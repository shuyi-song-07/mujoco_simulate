"""One frozen oblique view for snapshot replay, recording and future policies.

This module deliberately does not change the MuJoCo model or physical state.
The scene asset supplies the existing blue/black sky and blue/grey ground.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

TRAINING_CAMERA_NAME = 'oblique'
TRAINING_CAMERA_SPEC = {
    'version': 'task3-oblique-v1',
    'name': TRAINING_CAMERA_NAME,
    'projection': 'free_perspective',
    'distance': 1.4,
    'azimuth': 150.0,
    'elevation': -35.0,
    'lookat': [-0.1, -0.1, 0.5],
    'width': 640,
    'height': 480,
    'fps': 30,
}
LEGACY_CAMERAS = ('top', 'front', 'side')
OPERATOR_CAMERAS = ('top', 'front', TRAINING_CAMERA_NAME)
RECORDING_CAMERAS = (*LEGACY_CAMERAS, TRAINING_CAMERA_NAME)  # Historical files remain readable.


def training_camera_spec(config=None):
    """Validate the configured pose; keep the legacy frozen view reproducible."""
    specification = (config or {}).get('training_camera', TRAINING_CAMERA_SPEC)
    if not isinstance(specification, dict) or set(specification) != set(TRAINING_CAMERA_SPEC):
        raise ValueError('training_camera has missing or unknown fields')
    version = specification['version']
    if version not in ('task3-oblique-v1', 'papera-oblique-config-v1'):
        raise ValueError('Unsupported training_camera.version')
    if version == 'task3-oblique-v1' and specification != TRAINING_CAMERA_SPEC:
        raise ValueError('training_camera must match the frozen task3-oblique-v1 definition; '
                         'use papera-oblique-config-v1 for a new configurable view')
    for key in ('name', 'projection', 'width', 'height', 'fps'):
        if specification[key] != TRAINING_CAMERA_SPEC[key]:
            raise ValueError(f'training_camera.{key} must be {TRAINING_CAMERA_SPEC[key]}')
    for key in ('distance', 'azimuth', 'elevation', 'width', 'height', 'fps'):
        value = specification[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'training_camera.{key} must be a finite number')
    if specification['distance'] <= 0:
        raise ValueError('training_camera.distance must be positive (meters)')
    if not -90 <= specification['elevation'] <= 90:
        raise ValueError('training_camera.elevation must be within [-90, 90] degrees')
    lookat = specification['lookat']
    if (not isinstance(lookat, list) or len(lookat) != 3
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in lookat)):
        raise ValueError('training_camera.lookat must contain three finite coordinates (meters)')
    return copy.deepcopy(specification)


def make_training_camera(config=None):
    """Build a MuJoCo free camera without depending on XML camera IDs."""
    import mujoco
    specification = training_camera_spec(config)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = specification['lookat']
    for key in ('distance', 'azimuth', 'elevation'):
        setattr(camera, key, specification[key])
    return camera


def recording_camera_names(config=None):
    """Read the frozen choice: legacy three/four views or native oblique only."""
    names = (config or {}).get('recording', {}).get('cameras', LEGACY_CAMERAS)
    if (not isinstance(names, (list, tuple)) or not names or
            any(not isinstance(name, str) or name not in RECORDING_CAMERAS for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('recording.cameras must contain unique supported camera names')
    return tuple(names)


def validate_recording_cameras(config):
    names = recording_camera_names(config)
    if TRAINING_CAMERA_NAME in names:
        specification = training_camera_spec(config)
        for key in ('width', 'height', 'fps'):
            value = config['recording'].get(key)
            if type(value) not in (float, int) or value != specification[key]:
                raise ValueError(f'oblique recording requires {key}={specification[key]}')
        if names not in (OPERATOR_CAMERAS, RECORDING_CAMERAS, (TRAINING_CAMERA_NAME,)):
            raise ValueError('Oblique recording requires oblique only, top/front/oblique, or historical top/front/side/oblique')
    elif 'training_camera' in config:
        training_camera_spec(config)
    return names


def training_camera_metadata(config=None):
    specification = training_camera_spec(config)
    payload = json.dumps(specification, sort_keys=True, separators=(',', ':')).encode()
    return {
        **specification,
        'spec_sha256': hashlib.sha256(payload).hexdigest(),
        'background': {
            'source': 'simulation/mujoco/assets/dual_arm/dual_arm_scene.xml',
            'sky': {'builtin': 'gradient', 'rgb1': [0.3, 0.5, 0.7], 'rgb2': [0, 0, 0]},
            'ground': {'builtin': 'checker', 'rgb1': [0.2, 0.3, 0.4], 'rgb2': [0.1, 0.2, 0.3]},
        },
        'vertical_fov_source': 'model.vis.global_.fovy',
        'appearance': 'opaque robot; no operator-view transparency',
    }
