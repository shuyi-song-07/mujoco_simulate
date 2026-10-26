"""Tray task semantics shared by S1/S2/S3, without control or curation changes."""
import numpy as np
from simulation.mujoco.dual_arm.task_states import TaskState, ControlMode
from simulation.mujoco.dual_arm.panda_arm_controller import rotation_angle
from simulation.mujoco.dual_arm.rigid_control import rz
from simulation.mujoco.dual_arm.task_scenes import placement_geometry

def configure_xml(root, config):
    from simulation.mujoco.dual_arm.task_scenes import configure_paper_scene
    configure_paper_scene(root, config['scene'])

def reset(controller, object_position, yaw):
    from simulation.mujoco.dual_arm.task_scenes import reset_paper_obstacles
    return reset_paper_obstacles(controller.model, controller.config['scene'], object_position, yaw)

def rollout_monitor(config, initial_state, **options):
    from rollout.protocol import SuccessMonitor
    return SuccessMonitor(config, initial_state, **options)

def preflight(config, seeds, demonstration_seeds=(), engineering_seeds=()):
    from rollout.protocol import preflight_cases
    return preflight_cases(config, seeds, demonstration_seeds, engineering_seeds)

class TrayEvidenceAdapter:
    version = 'papera-tray-evidence-v1'
    def validate(self, metadata):
        if metadata.get('schema_version') != 'task3-papera-v2':
            raise ValueError('Tray evidence adapter requires task3-papera-v2')
    def build_evidence(self, episode, curation_config):
        from dataset.task3.frame_evidence import build_frame_evidence
        return build_frame_evidence(episode, curation_config)

def dataset_adapter():
    return TrayEvidenceAdapter()


def collection_success(self, c, now, object_pos, object_rot, any_contact):
    fully_open = all(a.gripper_command_width_m >= a.max_gripper_width_m-.001 for a in c.arms)
    e = c.config["evaluation"]
    at_target = np.linalg.norm(object_pos[:2]-c.target_position[:2]) <= e["target_position_tolerance_m"]
    velocity = c.data.qvel[c.object_dof_adr:c.object_dof_adr+6]
    settled = np.linalg.norm(velocity[:3]) <= e["stable_linear_speed_mps"] and np.linalg.norm(velocity[3:]) <= e["stable_angular_speed_radps"]
    good = (self.ever_dual and self.released and not any_contact and fully_open and self.support_ready and at_target
            and settled
            and abs(object_pos[2]-c.target_position[2]) <= e["target_height_tolerance_m"]
            and rotation_angle(object_rot, rz(c.target_yaw)) <= e["target_orientation_tolerance_rad"])
    if 'placement_platform' in c.config['scene']:
        good = good and self.support_surface == 'placement_platform' and placement_geometry(c)['contained']
    if good and c.task_state != TaskState.FAIL:
        if c._success_stable_since is None:
            c._success_stable_since = now
        if now-c._success_stable_since >= e["stable_success_seconds"]:
            c.task_state, c.control_mode = TaskState.DONE, ControlMode.IDLE
            c.metrics.completion_time_seconds = c.task_elapsed_seconds
    else:
        c._success_stable_since = None


def initial_state(config, seed):
    """Match DualArmTaskController.reset's random-number order exactly."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    scene = config["scene"]
    randomization = scene["randomization"]
    rng = np.random.default_rng(seed)
    position = np.asarray(scene["object_position"], dtype=float).copy()
    extent = np.asarray(randomization["object_xy_range_m"], dtype=float)
    position[:2] += rng.uniform(-extent, extent)
    position[2] = scene["table_top_z"] + scene["object_half_size"][2] + .0002
    yaw = float(rng.uniform(-randomization["object_yaw_range_rad"], randomization["object_yaw_range_rad"]))
    target = np.asarray(scene["target_position"], dtype=float).copy()
    extent = np.asarray(randomization["target_xy_range_m"], dtype=float)
    target[:2] += rng.uniform(-extent, extent)
    target[2] = scene["table_top_z"] + scene.get('placement_platform', {}).get('height_m', 0.) + scene["object_half_size"][2]
    return {"seed": seed, "object_position": position.tolist(), "object_yaw_rad": yaw,
            "object_quaternion_wxyz": [float(np.cos(yaw/2)), 0., 0., float(np.sin(yaw/2))],
            "target_position": target.tolist(), "target_yaw_rad": float(scene.get("target_yaw_rad", 0.))}

