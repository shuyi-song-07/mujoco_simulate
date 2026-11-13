"""Push-T rollout semantics: source coverage and a locked single gripper."""
import math
from .executor import SinglePushPolicyExecutor

def create_executor(controller, config=None):
    return SinglePushPolicyExecutor(controller, config)

def validate_success(settings):
    if settings['version']!='papera-single-push-t-evaluation-v1':
        raise ValueError('Unknown evaluation settings')
    open_fraction=settings.get('success_open_fraction',1.0)
    if (isinstance(open_fraction,bool) or not isinstance(open_fraction,(int,float))
            or not math.isfinite(open_fraction) or not 0 < open_fraction <= 1):
        raise ValueError('success_open_fraction must be in (0, 1]')
    if (settings.get('task_id')!='push_t' or settings.get('task_success_criterion')!='upstream-push-t-coverage-v1'
            or settings.get('success_coverage')!=.9 or settings.get('exit_coverage')!=.85 or settings.get('dwell_seconds')!=.5):
        raise ValueError('Push-T evaluation must retain the source coverage/dwell contract')

def validate_scene(settings, config):
    pass  # Source Push-T has no paired-grasp lift confirmation.

def success_description(default):
    return '单臂 Push-T 使用源任务的 T 形覆盖率：达到 90% 并连续保持 0.5 秒；夹爪全程闭合，不要求抓取、抬升或开爪。'

def update_stage_progress(controller, evidence, stages, elapsed):
    flags = {'push_contact': controller.contacts.get('robot_object', False),
             'coverage_90_percent': evidence.get('coverage', 0.) >= .9,
             'success': evidence.get('task_success', False)}
    for name, reached in flags.items():
        if reached and name not in stages:
            stages[name] = float(elapsed)

