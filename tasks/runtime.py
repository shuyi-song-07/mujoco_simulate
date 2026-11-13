"""Controller dispatch from an explicitly validated task schema."""
def create_controller(config, **kwargs):
    from tasks.data_schema import schema_for, SINGLE
    if schema_for(config) == SINGLE:
        from simulation.mujoco.single_arm.task_controller import SingleArmTaskController
        return SingleArmTaskController(config=config, **kwargs)
    from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
    return DualArmTaskController(config=config, **kwargs)


def create_model(config):
    from tasks.data_schema import schema_for, SINGLE
    if schema_for(config) == SINGLE:
        from tasks.push_t.scene import create_model as single_model
        return single_model(config)
    from simulation.mujoco.dual_arm.task_controller import create_model as dual_model
    return dual_model(config)
