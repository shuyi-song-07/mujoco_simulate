"""Collection wiring for the source-frozen single Panda Push-T implementation."""
controller_id = 'single_panda_push_v1'
TASK_IDS = ('push_t',)
page = 'single_arm.html'
backend_module = 'simulation.mujoco.common.teleop_service'
config_patch_scope = 'all'
schema = {"state": "single_panda_q7_yaw_width_m_s9", "action": "single_xyz_rpy_width_m_a7",
          "observation": "oblique_rgb", "state_dim": 9, "action_dim": 7}

def create_controller(config, **kwargs):
    from simulation.mujoco.single_arm.task_controller import SingleArmTaskController
    return SingleArmTaskController(config=config, **kwargs)

def create_model(config):
    from tasks.push_t.scene import create_model as factory
    return factory(config)
