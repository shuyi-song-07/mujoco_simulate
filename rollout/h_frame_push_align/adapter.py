"""h_frame_push_align uses shared dual-arm orchestration and its registered task monitor."""
from functools import partial
from rollout.common.push_task import validate_success as _validate, validate_scene, update_stage_progress, success_description
from tasks.h_frame_push_align.success import CRITERION
from .executor import create_executor

validate_success = partial(_validate, criterion=CRITERION)
