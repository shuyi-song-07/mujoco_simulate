"""Shared paired-arm evaluation checks, unchanged from the original runner."""
import math
from rollout.common.dual_executor import PolicyExecutor

def create_executor(controller, config=None):
    return PolicyExecutor(controller, config)

def validate_success(settings):
    from rollout.common.protocol import FULL_OPEN_SUCCESS_CRITERION, HALF_OPEN_SUCCESS_CRITERION
    if settings['version'] not in {'papera-closed-loop-evaluation-v1','papera-closed-loop-evaluation-v2','papera-single-view-half-open-evaluation-v1'}:
        raise ValueError('Unknown evaluation settings')
    open_fraction=settings.get('success_open_fraction',1.0)
    criterion=settings.get('success_criterion',FULL_OPEN_SUCCESS_CRITERION)
    if (isinstance(open_fraction,bool) or not isinstance(open_fraction,(int,float))
            or not math.isfinite(open_fraction) or not 0 < open_fraction <= 1):
        raise ValueError('success_open_fraction must be in (0, 1]')
    expected_criterion=HALF_OPEN_SUCCESS_CRITERION if open_fraction<1 else FULL_OPEN_SUCCESS_CRITERION
    if criterion!=expected_criterion:
        raise ValueError('success_criterion does not match success_open_fraction')
    if settings['version']=='papera-single-view-half-open-evaluation-v1' and open_fraction!=.5:
        raise ValueError('The current single-view baseline protocol freezes half-open success')

def validate_scene(settings, config):
    if settings['lift_stable_seconds'] != config['cooperation']['grasp_stable_seconds']:
        raise ValueError('Lift confirmation differs from the frozen success monitor')

def success_description(default):
    return default

def update_stage_progress(controller, evidence, stages, elapsed):
    """Diagnostic milestones only; never replace the physical success monitor."""
    flags = {
        "left_grasp": evidence.get("current_opposed_grasps", {}).get("left", False),
        "right_grasp": evidence.get("current_opposed_grasps", {}).get("right", False),
        "dual_grasp": evidence.get("ever_dual", False),
        "lift": evidence.get("lift_verified", False),
        "success": evidence.get("task_success", False),
    }
    if hasattr(controller, "arms"):
        import numpy as np
        c = controller
        actual = np.asarray([a.get_actual_pose()[0] for a in c.arms])
        position = c.data.xpos[c.object_id]
        rotation = c.data.xmat[c.object_id].reshape(3, 3)
        offsets = np.asarray(c.config["scene"]["grasp_offsets"])
        handles = position + offsets @ rotation.T
        flags["both_xy_aligned_30mm"] = bool(np.all(np.linalg.norm(actual[:, :2]-handles[:, :2], axis=1) <= .03))
        flags["both_near_grasp_height_30mm"] = bool(np.all(np.abs(actual[:, 2]-handles[:, 2]) <= .03))
        flags["any_finger_contact"] = any(any(c.contacts.get(side+"_fingers", [])) for side in ("left", "right"))
        flags["closed_below_45mm"] = all(arm.gripper_command_width_m <= .045 for arm in c.arms)
        flags["release_started"] = c.contact_control.release_count > 0
        flags["carry_xy_50mm"] = (flags["lift"] and
            np.linalg.norm(position[:2]-np.asarray(c.initial_state["object_position"][:2])) >= .05)
    for name, reached in flags.items():
        if reached and name not in stages:
            stages[name] = float(elapsed)

