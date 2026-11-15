"""push_r uses shared dual-arm orchestration and its registered task monitor."""
from functools import partial
from rollout.common.push_task import validate_success as _validate, validate_scene, update_stage_progress, success_description
from tasks.push_r.success import CRITERION
from .executor import create_executor

validate_success = partial(_validate, criterion=CRITERION)
