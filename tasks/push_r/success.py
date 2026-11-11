"""Push-R uses the existing planar pose-and-support alignment criterion."""
from tasks.h_frame_push_align.success import (
    PushAlignmentMonitor,
    collection_success as shared_collection_success,
)


CRITERION = 'papera-push-r-align-v1'


class PushRAlignmentMonitor(PushAlignmentMonitor):
    def update(self, controller):
        result = super().update(controller)
        result.update(version=CRITERION, success_criterion=CRITERION,
                      task_success_criterion=CRITERION)
        return result


def collection_success(contact_control, controller, now, object_pos, object_rot, any_contact):
    if getattr(contact_control, 'push_alignment_monitor', None) is None:
        contact_control.push_alignment_monitor = PushRAlignmentMonitor(controller.config)
    return shared_collection_success(contact_control, controller, now, object_pos, object_rot, any_contact)


def rollout_monitor(config, initial_state, **options):
    return PushRAlignmentMonitor(config)
