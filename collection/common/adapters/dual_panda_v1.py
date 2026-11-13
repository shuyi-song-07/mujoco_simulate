"""Collection wiring for the existing two-Panda implementation."""
controller_id = 'dual_panda_v1'
page = 'dual_arm.html'
backend_module = 'simulation.mujoco.common.teleop_service'
config_patch_scope = 'scene_only'
schema = {"state": "dual_panda_q7_width_m_s16", "action": "dual_xyz_rpy_width_m_a14",
          "observation": "oblique_rgb", "state_dim": 16, "action_dim": 14}

def create_controller(config, **kwargs):
    from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
    return DualArmTaskController(config=config, **kwargs)

def create_model(config):
    from simulation.mujoco.dual_arm.task_controller import create_model as factory
    return factory(config)
