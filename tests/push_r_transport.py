"""Push-R geometry, planar success and real Panda contact regression."""
import math
from pathlib import Path
import unittest

import mujoco
import numpy as np

from collection.config import recording_config
from rollout.executor import PolicyExecutor
from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
from simulation.mujoco.dual_arm.task_states import TaskState
from tasks.registry import ROOT, get, hook
from tests import h_frame_push_transport as reference
from tests.task3 import test_h_push_success as planar_tests


TASK = "push_r"


def push_demo(controller, capture=None):
    """Issue only ordinary gripper targets, keeping the R free in live physics."""
    c = controller
    c.reset(10)
    executor = PolicyExecutor(c)
    monitor = hook(c.config, "rollout_monitor")(c.config, c.initial_state)
    action = c.action_vector_m().copy()
    action[[6, 13]] = [a.max_gripper_width_m for a in c.arms]
    initial = c.data.xpos[c.object_id].copy()
    initial_yaw = c.get_status()["object_yaw_rad"]
    peak_force = np.zeros(2)
    max_lift = -math.inf
    min_width = math.inf
    ticks, phases = 0, []

    def move(label, positions, count):
        nonlocal ticks, max_lift, min_width
        action[:3], action[7:10] = [a.world_to_task(p) for a, p in zip(c.arms, positions)]
        for _ in range(count):
            applied = executor.apply(action)
            assert applied["accepted"], (label, applied["reason"])
            c.step()
            ticks += 1
            assert c.task_state != TaskState.FAIL, (label, c.failure_reason)
            assert not c.contact_control.ever_dual
            for i, side in enumerate(("left", "right")):
                peak_force[i] = max(peak_force[i], sum(c.contacts["finger_normal_force_n"][side]))
            max_lift = max(max_lift, c.contact_control.bottom_height_m)
            min_width = min(min_width, *(a.gripper_status()["actual_width_m"] for a in c.arms))
            result = monitor.update(c)
            if capture and ticks % 2 == 0:
                capture(None)
            if result["task_success"]:
                break
        phases.append({"phase": label, "time_s": float(c.data.time),
                       "object_position": c.data.xpos[c.object_id].tolist(),
                       "object_yaw_rad": c.get_status()["object_yaw_rad"]})
        if capture:
            capture(label)
        return result

    if capture:
        capture("initial")
    waypoints = np.array([a.get_actual_pose()[0] for a in c.arms])
    move("settle", waypoints, 30)
    np.testing.assert_allclose(c.data.xpos[c.object_id, :2], initial[:2], atol=.001)
    rear = c.data.xpos[c.object_id] + np.asarray(c.config["scene"]["grasp_offsets"]) @ c.data.xmat[c.object_id].reshape(3, 3).T
    waypoints = rear.copy()
    waypoints[:, 0] -= .063
    waypoints[:, 2] = .49
    move("approach", waypoints, 100)
    waypoints[:, 2] = c.config["scene"]["workspace_z_min_m"]
    move("descend", waypoints, 85)
    assert not peak_force.any(), "Approach must stop before contact"
    waypoints[:, 0] += [.169, .135]
    move("dual_push", waypoints, 135)
    assert c.data.xpos[c.object_id, 0] - initial[0] > .09
    assert np.all(peak_force > 1), "Both arms must physically push"
    waypoints[:, 2] = .49
    move("retract", waypoints, 65)
    # The forward top bar gives the opposite yaw correction to the rear spine.
    rim = c.data.xpos[c.object_id] + c.data.xmat[c.object_id].reshape(3, 3) @ [.09, .15, 0]
    waypoints[1, :2] = rim[:2] + [0, .026]
    move("side_approach", waypoints, 90)
    waypoints[1, 2] = c.config["scene"]["workspace_z_min_m"]
    move("side_descend", waypoints, 85)
    waypoints[1, 1] -= .075
    result = move("aligned", waypoints, 100)
    assert c.task_state == TaskState.DONE and result["task_success"], result
    assert abs(math.remainder(c.get_status()["object_yaw_rad"] - initial_yaw, math.tau)) > .05
    assert max_lift < .002 and min_width >= .070
    assert c.metrics.collision_count == 0
    return {"input_kind": "scripted_test", "seed": 10, "phases": phases,
            "peak_fingertip_normal_force_n": peak_force.tolist(),
            "max_bottom_clearance_m": max_lift, "minimum_measured_aperture_m": min_width,
            "final": result}


class PushRTests(planar_tests.HFrameSuccessTests):
    @classmethod
    def setUpClass(cls):
        cls.controller = DualArmTaskController(get(TASK).config())

    def test_inherited_table_robot_and_camera(self):
        old, new = [DualArmTaskController(recording_config(get(task)))
                    for task in ("h_frame_push_align", TASK)]
        reference.verify_inheritance(old, new)
        self.assertAlmostEqual(new.model.body_mass[new.object_id], .55)
        self.assertTrue(np.all(new.model.body_inertia[new.object_id] > 0))
        self.assertEqual(new.model.neq, old.model.neq)
        goal = np.flatnonzero(new.model.geom_bodyid == new.target_id)
        self.assertFalse(new.model.geom_contype[goal].any())
        self.assertFalse(new.model.geom_conaffinity[goal].any())

    def test_task_identity_and_dataset_adapter(self):
        import json
        from rollout.rollout import validate_settings

        config = self.c.config
        settings = json.loads((ROOT / get(TASK).manifest["evaluation_config"]).read_text(encoding="utf-8"))
        validate_settings(settings)
        adapter = hook(config, "dataset_adapter")()
        adapter.validate({"task_id": TASK, "schema_version": "task3-papera-v2"})
        with self.assertRaises(ValueError):
            adapter.validate({"task_id": "h_frame_push_align", "schema_version": "task3-papera-v2"})
        result = self.monitor.update(self.c)
        self.assertEqual(result["task_success_criterion"], "papera-push-r-align-v1")

    def test_reproducible_nonterminal_resets(self):
        for seed in range(20):
            self.c.reset(seed)
            state = self.c.data.qpos.copy()
            self.c.reset(seed)
            np.testing.assert_array_equal(self.c.data.qpos, state)
            monitor = hook(self.c.config, "rollout_monitor")(self.c.config, self.c.initial_state)
            for _ in range(30):
                self.c.step()
            self.assertFalse(monitor.update(self.c)["task_success"])
            self.assertNotEqual(self.c.task_state, TaskState.FAIL)

    def test_real_contact_translation_and_yaw(self):
        push_demo(DualArmTaskController(get(TASK).config()))


def render_demo(output):
    import json
    from PIL import Image
    from rollout.rollout import PanelVideo, decode_video
    from simulation.mujoco.dual_arm.render_workers import MultiCameraRenderer

    output = output.resolve()
    if not output.is_relative_to(ROOT / ".local/checks"):
        raise ValueError("Write engineering media below .local/checks")
    output.mkdir(parents=True, exist_ok=False)
    c = DualArmTaskController(recording_config(get(TASK)))
    renderer = MultiCameraRenderer(c.model, config=c.config, camera_names=("top", "oblique"))
    video = PanelVideo(output / "contact_push.mp4", 30, 23, cameras=("oblique",))

    def capture(label):
        if label is None:
            video.append(renderer.render(c.data))
        elif label in ("initial", "dual_push", "aligned"):
            for name, image in renderer.render(c.data).items():
                Image.fromarray(image).save(output / f"{label}_{name}.png")

    try:
        result = push_demo(c, capture)
    finally:
        video.close()
        renderer.close()
    result["decoded_video"] = decode_video(video.path, video.frames, width=640)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", type=Path, required=True)
    print(json.dumps(render_demo(parser.parse_args().media), indent=2))
