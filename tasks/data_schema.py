"""Explicit robot layouts shared by collection and offline data validation.

The legacy dual-arm layout remains the default for historical artifacts. A
single arm is never represented by a padded/fabricated second arm. The scoped
layout is context-local, so concurrent episode audits cannot affect each other.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import json
from pathlib import Path


@dataclass(frozen=True)
class DataSchema:
    version: str
    arm_order: tuple[str, ...]
    joints_per_arm: int

    @property
    def state_dim(self): return len(self.arm_order) * (self.joints_per_arm + 1)
    @property
    def action_dim(self): return len(self.arm_order) * 7
    @property
    def state_stride(self): return self.joints_per_arm + 1
    @property
    def state_names(self):
        return [
            name for side in self.arm_order for name in
            ([f'{side}_joint{i+1}' for i in range(7)] +
             ([f'{side}_gripper_yaw_rad'] if self.joints_per_arm == 8 else []) +
             [f'{side}_finger_width'])]
    @property
    def action_names(self):
        return [f'{side}_{name}' for side in self.arm_order for name in
                ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper_width_m')]


DUAL = DataSchema('task3-papera-v2', ('left', 'right'), 7)
SINGLE = DataSchema('single-panda-push-v1', ('left',), 8)
_ACTIVE = ContextVar('papera_data_schema', default=DUAL)


def schema_for(value=None):
    """Resolve only declared layouts; never guess a robot from vector length."""
    if isinstance(value, DataSchema): return value
    if value is None: return _ACTIVE.get()
    if isinstance(value, (list, tuple)):
        return schema_for(value[0]) if value else _ACTIVE.get()
    if not isinstance(value, dict): return _ACTIVE.get()
    version = value.get('schema_version')
    if version in (DUAL.version, SINGLE.version):
        result = SINGLE if version == SINGLE.version else DUAL
    elif 'data_schema' in value:
        result = schema_for(value['data_schema'])
    elif isinstance(value.get('schema'), dict):
        schema = value['schema']
        if schema.get('state') == 'single_panda_q7_yaw_width_m_s9': result = SINGLE
        elif schema.get('state') == 'dual_panda_q7_width_m_s16': result = DUAL
        else: raise ValueError('Unsupported declared robot data schema')
        expected_action = ('single_xyz_rpy_width_m_a7' if result == SINGLE else 'dual_xyz_rpy_width_m_a14')
        if (schema.get('action') != expected_action or schema.get('state_dim') != result.state_dim
                or schema.get('action_dim') != result.action_dim or schema.get('observation') != 'oblique_rgb'):
            raise ValueError('Declared robot schema has incompatible action, dimensions or camera')
    elif 'schema_version' in value:
        raise ValueError('Unsupported declared robot schema version')
    else:
        result = _ACTIVE.get()
        for key in ('metadata', 'task_binding', 'config'):
            if isinstance(value.get(key), dict):
                result = schema_for(value[key])
                break
    if 'arm_order' in value and tuple(value['arm_order']) != result.arm_order:
        raise ValueError('Arm order disagrees with declared robot data schema')
    for key in ('state_dim', 'action_dim'):
        if key in value and value[key] != getattr(result, key):
            raise ValueError(f'{key} disagrees with declared robot data schema')
    if 'hand_order' in value and tuple(value['hand_order']) != result.arm_order:
        raise ValueError('Hand order disagrees with declared robot data schema')
    if isinstance(value.get('task_binding'), dict) and schema_for(value['task_binding']) != result:
        raise ValueError('Task binding disagrees with declared robot data schema')
    return result


def descriptor(value=None):
    s = schema_for(value)
    return {'schema_version': s.version, 'arm_order': list(s.arm_order),
            'state_dim': s.state_dim, 'action_dim': s.action_dim}


@contextmanager
def schema_context(value):
    token = _ACTIVE.set(schema_for(value))
    try: yield _ACTIVE.get()
    finally: _ACTIVE.reset(token)


def scoped(function):
    """Bind layout for a public episode/row operation, restoring it afterwards."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        def declared(value):
            if isinstance(value, (str, Path)):
                p = Path(value)
                if p.is_dir() and (p / 'manifest.json').is_file():
                    return declared(json.loads((p / 'manifest.json').read_text(encoding='utf-8')))
                return None
            if isinstance(value, (list, tuple)):
                return declared(value[0]) if value else None
            if not isinstance(value, dict): return None
            if ('schema_version' in value or 'data_schema' in value
                    or isinstance(value.get('schema'), dict) or isinstance(value.get('task_binding'), dict)):
                return value
            for key in ('data_schema', 'metadata', 'task_binding', 'config', 'source_ref', 'rows'):
                found = declared(value.get(key))
                if found is not None: return found
            if value.get('metadata_path'):
                return declared(json.loads(Path(value['metadata_path']).read_text(encoding='utf-8')))
            return None
        value = next((found for v in (*args, *kwargs.values())
                      if (found := declared(v)) is not None), None)
        with schema_context(value): return function(*args, **kwargs)
    return wrapped


class _ArmOrder:
    def __iter__(self): return iter(schema_for().arm_order)
    def __len__(self): return len(schema_for().arm_order)
    def __getitem__(self, index): return schema_for().arm_order[index]


SIDES = _ArmOrder()
def state_dim(): return schema_for().state_dim
def action_dim(): return schema_for().action_dim
def state_stride(): return schema_for().state_stride
def joint_count(): return schema_for().joints_per_arm
def state_width_indices(): return [j * state_stride() + joint_count() for j in range(len(SIDES))]
def action_width_indices(): return [j * 7 + 6 for j in range(len(SIDES))]
def angle_columns(): return [j * 7 + k for j in range(len(SIDES)) for k in (3, 4, 5)]
