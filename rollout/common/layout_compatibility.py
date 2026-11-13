"""Narrow source-only compatibility for frozen evaluation records."""
from pathlib import Path

from common.layout import ROOT, equivalent_binding, migration_reason, source_path


def task_config_matches(recorded, current):
    if recorded == current:
        return True
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        return False
    old, new = dict(recorded), dict(current)
    a, b = old.pop('task_binding', None), new.pop('task_binding', None)
    return old == new and equivalent_binding(a, b)


def _catalog(value):
    result = {}
    for name, digest in value.items():
        path = source_path(name)
        key = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        if key in result:
            raise ValueError('Duplicate canonical provenance source')
        result[key] = digest
    return result


def policy_sources_match(recorded, current):
    """Only checked source bytes and their explicit compatibility label may vary.

    Checkpoint, data, model, camera, timing, precision and all other fields stay
    exact. File hashes in either record remain their actual measured hashes.
    """
    if recorded == current:
        return True
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        return False
    old, new = dict(recorded), dict(current)
    before, after = old.pop('checked_hashes', None), new.pop('checked_hashes', None)
    old_mode, new_mode = old.pop('source_compatibility', None), new.pop('source_compatibility', None)
    if old != new or not isinstance(before, dict) or not isinstance(after, dict):
        return False
    before, after = _catalog(before), _catalog(after)
    if before.keys() != after.keys() or not all(
            digest == after[name] or migration_reason(name, digest, after[name])
            for name, digest in before.items()):
        return False
    if old_mode == new_mode:
        return True
    if new_mode != 'reviewed-training-interface-only':
        return False
    # A changed label is justified only by a certified changed source pair,
    # never merely by a string value or by unchanged data/checkpoint files.
    return any(migration_reason(name, digest, after[name])
        for name, digest in before.items())
