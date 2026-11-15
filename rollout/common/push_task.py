"""Non-grasp planar alignment diagnostics; physical success stays in the task."""


def validate_success(settings, *, criterion):
    if settings.get('version') != 'papera-push-alignment-evaluation-v1':
        raise ValueError('Push alignment requires its task-specific evaluation settings')
    if settings.get('success_criterion') != criterion or settings.get('task_success_criterion') != criterion:
        raise ValueError('Push alignment success criterion differs from the task monitor')


def validate_scene(settings, config):
    if not config['scene'].get('push_alignment'):
        raise ValueError('Push alignment scene parameters are missing')


def success_description(default):
    return '按任务配置判断物体 XY、Yaw、桌面支撑和稳定持续时间；不要求双抓、抬升或开爪。'


def update_stage_progress(controller, evidence, stages, elapsed):
    conditions = evidence.get('final_conditions', {})
    flags = {side + '_push_contact': any(controller.contacts.get(side + '_fingers', []))
             for side in ('left', 'right')}
    flags.update(position_aligned=conditions.get('position_xy', False),
                 yaw_aligned=conditions.get('yaw', False),
                 success=evidence.get('task_success', False))
    for name, reached in flags.items():
        if reached and name not in stages:
            stages[name] = float(elapsed)
