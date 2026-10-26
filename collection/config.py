"""Capture output options, independent of the registered physical task version."""
import json
import math
from tasks.registry import ROOT
from simulation.mujoco.dual_arm.training_camera import validate_recording_cameras


def recording_config(task):
    config = task.config()
    options = json.loads((ROOT / 'config/recording.json').read_text(encoding='utf-8'))
    if not isinstance(options, dict) or set(options) - {'cameras', 'post_success_seconds', '_help'} or 'cameras' not in options:
        raise ValueError('recording.json supports cameras, post_success_seconds and _help')
    tail = options.get('post_success_seconds', 0.)
    if isinstance(tail, bool) or not isinstance(tail, (float, int)) or not math.isfinite(tail) or not 0 <= tail <= 2:
        raise ValueError('post_success_seconds must be a finite number in [0, 2] seconds')
    config['recording']['post_success_seconds'] = float(tail)
    config['recording']['cameras'] = options['cameras']
    validate_recording_cameras(config)
    # The full resulting config and its hash are frozen in every recording.
    # Do not alter task_binding: output view selection is not a physics change.
    return config
