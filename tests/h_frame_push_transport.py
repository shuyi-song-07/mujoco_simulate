"""Validate inherited scene geometry and a real open-jaw H-frame push."""
import argparse
import json
import math
from pathlib import Path
import unittest

import mujoco
import numpy as np

from collection.config import recording_config
from rollout.executor import PolicyExecutor
from simulation.mujoco.dual_arm.action_contract import V2
from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
from simulation.mujoco.dual_arm.task_states import TaskState
from simulation.mujoco.dual_arm.training_camera import make_training_camera, training_camera_spec
from tasks.registry import ROOT, get, hook


TASK = 'h_frame_push_align'
DEMO_SEED = 10


def verify_inheritance(old, new):
    """Compare the compiled model and reset state, not just JSON parameters."""
    for name in ('table_geom', 'left_pedestal_geom', 'right_pedestal_geom'):
        a, b = old.model.geom(name), new.model.geom(name)
        for field in ('size', 'rgba', 'friction', 'pos', 'quat'):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        np.testing.assert_array_equal(old.data.geom(name).xpos, new.data.geom(name).xpos)
        np.testing.assert_array_equal(old.data.geom(name).xmat, new.data.geom(name).xmat)
    for field in ('mat_rgba', 'mat_specular', 'mat_shininess', 'mat_reflectance'):
        a, b = old.model.geom('table_geom').matid[0], new.model.geom('table_geom').matid[0]
        np.testing.assert_array_equal(getattr(old.model, field)[a], getattr(new.model, field)[b])
    for a, b in zip(old.arms, new.arms):
        name = f'{a.side}_link0'
        np.testing.assert_array_equal(old.data.body(name).xpos, new.data.body(name).xpos)
        np.testing.assert_array_equal(old.data.body(name).xquat, new.data.body(name).xquat)
        np.testing.assert_array_equal(old.data.qpos[a.qpos_indices], new.data.qpos[b.qpos_indices])
        np.testing.assert_array_equal(old.model.jnt_range[a.joint_ids], new.model.jnt_range[b.joint_ids])
    for field in ('actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange', 'tex_data'):
        np.testing.assert_array_equal(getattr(old.model, field), getattr(new.model, field))
    for field in ('cam_xpos', 'cam_xmat'):
        np.testing.assert_array_equal(getattr(old.data, field), getattr(new.data, field))
    for field in ('cam_fovy', 'cam_projection'):
        np.testing.assert_array_equal(getattr(old.model, field), getattr(new.model, field))
    assert old.model.opt.timestep == new.model.opt.timestep
    assert old.config['scene']['control_hz'] == new.config['scene']['control_hz']
    assert old.config['recording'] == new.config['recording']
    assert old.config['task_binding']['schema'] == new.config['task_binding']['schema']
    assert training_camera_spec(old.config) == training_camera_spec(new.config)
    cameras = []
    for c in (old, new):
        view = mujoco.MjvScene(c.model, maxgeom=1000)
        mujoco.mjv_updateScene(c.model, c.data, mujoco.MjvOption(), None,
                              make_training_camera(c.config), mujoco.mjtCatBit.mjCAT_ALL, view)
        camera = mujoco.mjv_averageCamera(*view.camera)
        cameras.append({name: np.asarray(getattr(camera, name)).tolist() for name in
                        ('pos', 'forward', 'up', 'frustum_top', 'frustum_bottom', 'frustum_center',
                         'frustum_near', 'frustum_far', 'orthographic')})
    assert cameras[0] == cameras[1]
    assert old.model.vis.global_.fovy == new.model.vis.global_.fovy
    return cameras[1]


def push_demo(c, capture=None):
    """Send ordinary arm targets. No object pose, velocity or force is written."""
    c.reset(DEMO_SEED)
    executor = PolicyExecutor(c)
    monitor = hook(c.config, 'rollout_monitor')(c.config, c.initial_state)
    action = c.action_vector_m().copy()
    action[[6, 13]] = [arm.max_gripper_width_m for arm in c.arms]
    first = c.data.xpos[c.object_id].copy()
    first_yaw = c.get_status()['object_yaw_rad']
    peak_force = np.zeros(2)
    max_bottom = -math.inf
    min_aperture = math.inf
    ticks = 0
    phases = []

    def move(label, positions, count):
        nonlocal ticks, max_bottom, min_aperture
        action[:3], action[7:10] = [arm.world_to_task(p) for arm, p in zip(c.arms, positions)]
        for _ in range(count):
            applied = executor.apply(action)
            assert applied['accepted'], (label, applied['reason'])
            c.step(1/60)
            ticks += 1
            assert c.task_state != TaskState.FAIL, (label, c.failure_reason)
            assert not c.contact_control.ever_dual, label
            for i, side in enumerate(('left', 'right')):
                peak_force[i] = max(peak_force[i], sum(c.contacts['finger_normal_force_n'][side]))
            max_bottom = max(max_bottom, c.contact_control.bottom_height_m)
            min_aperture = min(min_aperture, *(arm.gripper_status()['actual_width_m'] for arm in c.arms))
            result = monitor.update(c)
            if capture and ticks % 2 == 0:
                capture(None)
            if c.task_state == TaskState.DONE:
                break
        status = c.get_status()
        phases.append({'phase': label, 'time_s': status['simulation_time'],
                       'position_m': status['object_position'], 'yaw_rad': status['object_yaw_rad']})
        if capture:
            capture(label)
        return result

    if capture:
        capture('initial')
    neutral = np.array([arm.get_actual_pose()[0] for arm in c.arms])
    move('settle', neutral, 30)
    assert np.linalg.norm(c.data.xpos[c.object_id, :2] - first[:2]) < .001
    assert c.contact_control.support_ready
    rear = c.data.xpos[c.object_id] + np.asarray(c.config['scene']['grasp_offsets']) @ c.data.xmat[c.object_id].reshape(3, 3).T
    waypoints = rear.copy()
    waypoints[:, 0] -= .063
    waypoints[:, 2] = .49
    move('approach', waypoints, 100)
    waypoints[:, 2] = .367
    move('descend', waypoints, 85)
    assert np.all(peak_force == 0), 'Approach must stop before contact'
    waypoints[:, 0] += [.172, .138]
    move('dual_push', waypoints, 135)
    assert c.data.xpos[c.object_id, 0] - first[0] > .09
    assert np.all(peak_force > 1), 'Each arm must apply measurable fingertip contact'
    waypoints[:, 2] = .49
    move('retract', waypoints, 65)
    # Move the right arm outside the rear portion of its rail. A -Y push
    # there moves the frame toward the centerline and creates positive yaw.
    rail = c.data.xpos[c.object_id] + c.data.xmat[c.object_id].reshape(3, 3) @ [-.081, .16, 0]
    waypoints[1, :2] = rail[:2] + [0, .026]
    move('side_approach', waypoints, 90)
    waypoints[1, 2] = .367
    move('side_descend', waypoints, 85)
    waypoints[1, 1] -= .045
    result = move('aligned', waypoints, 100)
    assert c.task_state == TaskState.DONE and result['task_success'], result
    assert c.get_status()['object_yaw_rad'] - first_yaw > math.radians(8)
    assert max_bottom < .002 and min_aperture >= .070
    assert c.metrics.collision_count == 0
    return {'seed': DEMO_SEED, 'phases': phases, 'peak_fingertip_normal_force_n': peak_force.tolist(),
            'max_bottom_clearance_m': max_bottom, 'minimum_measured_aperture_m': min_aperture,
            'final': result}


class HFrameTransportTests(unittest.TestCase):
    def test_compiled_inheritance(self):
        old, new = [DualArmTaskController(recording_config(get(task))) for task in ('tray_to_platform', TASK)]
        verify_inheritance(old, new)
        target = np.flatnonzero(new.model.geom_bodyid == new.target_id)
        self.assertTrue(len(target) and not new.model.geom_contype[target].any()
                        and not new.model.geom_conaffinity[target].any())
        self.assertEqual(len(new.object_geoms), 3)
        self.assertAlmostEqual(new.model.body_mass[new.object_id], .55)

    def test_scripted_real_contact_alignment(self):
        push_demo(DualArmTaskController(get(TASK).config()))

    def test_z_floor_stops_descent_without_blocking_xy(self):
        config = get(TASK).config()
        config["scene"]["workspace_limits_enabled"] = 0
        c = DualArmTaskController(config)
        protocol = c.contact_control.control_protocol
        for _ in range(130):
            action = c.action_vector_m()
            action[[2, 9]] -= .002
            result = c.apply_action(
                action, schema_version=V2, components=("motion",), mode="Z",
                control_protocol=protocol, interaction_modes=("Z_OPEN", "Z_OPEN"),
                release_intent=(True, True),
            )
            if not result["accepted"]:
                break
            c.step(1 / 60)
        self.assertEqual(result["reason"], "workspace_limit")
        floor = config["scene"]["workspace_z_min_m"]
        np.testing.assert_allclose(c.action_vector_m()[[2, 9]], floor, atol=1e-5)
        action = c.action_vector_m()
        action[[0, 7]] += .002
        result = c.apply_action(
            action, schema_version=V2, components=("motion",), mode="XY",
            control_protocol=protocol, interaction_modes=("XY_HOLD", "XY_HOLD"),
        )
        self.assertTrue(result["accepted"], result["reason"])

    def test_workspace_switch_restores_y_boundary(self):
        config = get(TASK).config()
        old_limit = .060
        config["scene"]["right_workspace"][1][0] = old_limit
        target_y = old_limit - .005
        for enabled in (0, 1):
            with self.subTest(enabled=enabled):
                config["scene"]["workspace_limits_enabled"] = enabled
                c = DualArmTaskController(config)
                for _ in range(130):
                    action = c.action_vector_m()
                    action[8] = max(target_y, action[8] - .002)
                    result = c.apply_action(
                        action, schema_version=V2, components=("motion",), mode="XY",
                        control_protocol=c.contact_control.control_protocol,
                        interaction_modes=("HOLD", "XY_HOLD"),
                    )
                    if not result["accepted"]:
                        break
                    c.step(1 / 60)
                self.assertEqual(result["accepted"], not enabled, result["reason"])
                if enabled:
                    self.assertEqual(result["reason"], "workspace_limit")
                else:
                    self.assertLess(c.right.get_actual_pose()[0][1], old_limit)
                right_y = c.right.get_command_pose()[0][1]
                self.assertAlmostEqual(right_y, old_limit if enabled else target_y, places=5)


def render_demo(output):
    from PIL import Image
    from rollout.rollout import PanelVideo, decode_video
    from simulation.mujoco.dual_arm.render_workers import MultiCameraRenderer

    output = output.resolve()
    if not output.is_relative_to(ROOT / '.local/checks'):
        raise ValueError('Write engineering media below .local/checks')
    c = DualArmTaskController(recording_config(get(TASK)))
    old = DualArmTaskController(recording_config(get('tray_to_platform')))
    camera = verify_inheritance(old, c)
    output.mkdir(parents=True, exist_ok=False)
    renderer = MultiCameraRenderer(c.model, config=c.config, camera_names=('top', 'oblique'))
    video = PanelVideo(output / 'contact_push.mp4', 30, 23, cameras=('oblique',))

    def capture(phase):
        if phase is None:
            video.append(renderer.render(c.data))
        elif phase in ('initial', 'dual_push', 'aligned'):
            images = renderer.render(c.data)
            for name, image in images.items():
                Image.fromarray(image).save(output / f'{phase}_{name}.png')

    try:
        result = push_demo(c, capture)
    finally:
        video.close()
        renderer.close()
    result.update(oblique_camera=camera, decoded_video=decode_video(video.path, video.frames, width=640),
                  input_kind='scripted_test')
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media', type=Path, required=True)
    print(json.dumps(render_demo(parser.parse_args().media), indent=2))
