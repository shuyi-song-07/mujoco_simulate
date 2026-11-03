"""Explicit, scoped access to compatible frozen collection configurations.

This scope never changes a task binding. Callers must first compare the frozen
episode configurations and supply the checked differences and metadata paths.
Only the collection configuration's source digest may differ from the current
task; every other frozen task field and source remains an exact match.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

SOURCE = "config/collection.json"
PROOF_FIELD = "recorded_collection_compatibility"
VERSION = "papera-recorded-collection-compatibility-v1"
EQUIVALENCE_FIELD = "configuration_equivalence"
EQUIVALENCE_VERSION = "papera-recorded-config-equivalence-v1"
ALLOWED_CONFIG_PATHS = frozenset({
    "gesture_control.close_score_min",
    "gesture_control.open_score_min",
    "gesture_control.left_close_score_min",
})


@dataclass(frozen=True)
class _Scope:
    binding: dict
    current_binding: dict
    provenance: dict


_ACTIVE: ContextVar[_Scope | None] = ContextVar("recorded_collection_scope", default=None)


def config_differences(current, recorded, prefix=""):
    """Compare JSON values without equating booleans with numeric settings."""
    if isinstance(current, dict) and isinstance(recorded, dict):
        result = []
        for key in sorted(current.keys() | recorded.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in current or key not in recorded:
                result.append({"path": path, "current": current.get(key), "recorded": recorded.get(key)})
            else:
                result.extend(config_differences(current[key], recorded[key], path))
        return result
    if isinstance(current, list) and isinstance(recorded, list) and len(current) == len(recorded):
        return [difference for index, (left, right) in enumerate(zip(current, recorded))
                for difference in config_differences(left, right, f"{prefix}[{index}]")]
    numeric = type(current) in (int, float) and type(recorded) in (int, float)
    if (type(current) is type(recorded) or numeric) and current == recorded:
        return []
    return [{"path": prefix, "current": current, "recorded": recorded}]


def configuration_equivalence(reference, recorded, reference_metadata_path=None):
    """Describe complete configurations; scope entry independently checks them."""
    return {"version": EQUIVALENCE_VERSION,
            "reference_collection_config": copy.deepcopy(reference),
            "frozen_collection_config": copy.deepcopy(recorded),
            "reference_metadata_path": (str(Path(reference_metadata_path).resolve())
                                        if reference_metadata_path is not None else None)}


def _metadata_matches(path, binding, config):
    try:
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("Equivalence metadata paths must be absolute")
        metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read configuration equivalence metadata") from error
    if (not isinstance(metadata, dict) or metadata.get("task_id") != binding["task_id"]
            or metadata.get("task_binding") != binding
            or metadata.get("scene_id") != binding.get("scene_id")
            or not isinstance(metadata.get("config"), dict)
            or metadata["config"].get("task_binding") != binding
            or config_differences(metadata["config"], config)):
        raise ValueError("Configuration equivalence metadata task/config/binding differs")


def _validate_equivalence(proof, binding, current, task):
    evidence = proof.get(EQUIVALENCE_FIELD)
    if not isinstance(evidence, dict) or evidence.get("version") != EQUIVALENCE_VERSION:
        raise ValueError("Empty config_differences require complete configuration equivalence evidence")
    reference = evidence.get("reference_collection_config")
    recorded = evidence.get("frozen_collection_config")
    if (not isinstance(reference, dict) or reference.get("task_binding") != current
            or not isinstance(recorded, dict) or recorded.get("task_binding") != binding):
        raise ValueError("Configuration equivalence reference/recorded binding differs")
    left, right = copy.deepcopy(reference), copy.deepcopy(recorded)
    left.pop("task_binding")
    right.pop("task_binding")
    if config_differences(left, right):
        raise ValueError("Configuration equivalence scene/camera/settings differ")
    reference_path = evidence.get("reference_metadata_path")
    if reference_path is not None:
        _metadata_matches(reference_path, current, reference)
    else:
        from collection.config import recording_config
        if task is None or config_differences(recording_config(task), reference):
            raise ValueError("Configuration equivalence differs from the current recording configuration")
    for path in proof["metadata_paths"]:
        _metadata_matches(path, binding, recorded)


def _collection_only(binding, current):
    """Compare full bindings, permitting one well-formed source digest only."""
    if not isinstance(binding, dict) or not isinstance(binding.get("sources"), dict):
        return False
    sources = binding["sources"]
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
           for value in (sources.get(SOURCE), current["sources"].get(SOURCE))):
        return False
    normalized = copy.deepcopy(binding)
    normalized["sources"][SOURCE] = current["sources"][SOURCE]
    return normalized == current


def _proof(provenance, binding, current, task=None):
    if not isinstance(provenance, dict):
        raise ValueError("Recorded collection compatibility requires explicit provenance")
    # JSON round-tripping freezes caller-owned values and excludes non-finite
    # values or runtime objects from a durable artifact's provenance.
    try:
        proof = json.loads(json.dumps(provenance, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Recorded collection provenance must be finite JSON") from error
    differences = proof.get("config_differences")
    if not isinstance(differences, list):
        raise ValueError("Recorded collection provenance requires config_differences")
    paths = []
    for difference in differences:
        if not isinstance(difference, dict) or difference.get("path") not in ALLOWED_CONFIG_PATHS:
            raise ValueError("Only recorded gesture confidence thresholds may differ")
        path = difference["path"]
        if path in paths or set(difference) != {"path", "current", "recorded"}:
            raise ValueError("Each recorded configuration difference must occur exactly once")
        paths.append(path)
        for name in ("current", "recorded"):
            value = difference[name]
            if value is None and path == "gesture_control.left_close_score_min":
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Recorded gesture thresholds must be confidence fractions in [0, 1]")
        if difference["current"] == difference["recorded"]:
            raise ValueError("Recorded configuration differences must describe changed values")
    metadata = proof.get("metadata_paths")
    if (not isinstance(metadata, list) or not metadata or
            any(not isinstance(path, str) or not Path(path).is_absolute() for path in metadata)):
        raise ValueError("Recorded collection provenance requires absolute metadata_paths")
    if not differences:
        _validate_equivalence(proof, binding, current, task)
    proof.update({
        "version": VERSION,
        "task_id": binding["task_id"],
        "source_path": SOURCE,
        "recorded_collection_sha256": binding["sources"][SOURCE],
        "current_collection_sha256": current["sources"][SOURCE],
    })
    return proof


@contextmanager
def recorded_collection_scope(binding, provenance):
    """Allow one exact recorded binding while preserving default strict checks.

    ``provenance`` contains ``config_differences`` (a list of dictionaries with
    ``path``, ``current`` and ``recorded`` values) and ``metadata_paths`` (the
    absolute paths of the checked episode metadata). A missing optional left
    threshold is represented by ``None``. Current bindings need no exception
    and make this scope a no-op. Empty differences additionally require complete
    configuration-equivalence evidence, checked against actual metadata and a
    current metadata/configuration reference. The yielded proof is an independent
    copy; source hashes are never rewritten.
    """
    from tasks.registry import get

    if not isinstance(binding, dict) or not binding.get("task_id"):
        raise ValueError("Recorded collection scope requires a complete task binding")
    task = get(binding["task_id"])
    current = task.binding()
    if binding == current:
        yield None
        return
    if not _collection_only(binding, current):
        raise ValueError("Recorded collection scope permits only config/collection.json source differences")
    proof = _proof(provenance, binding, current, task)
    active = _Scope(copy.deepcopy(binding), copy.deepcopy(current), proof)
    token = _ACTIVE.set(active)
    try:
        yield copy.deepcopy(proof)
    finally:
        _ACTIVE.reset(token)


def accepts_recorded_collection(binding, current_binding):
    """Registry hook: accept only the scope's exact old and current bindings."""
    active = _ACTIVE.get()
    return bool(active is not None and binding == active.binding and
                current_binding == active.current_binding)


def recorded_collection_provenance(binding):
    """Return the active proof without exposing the stored mutable objects."""
    active = _ACTIVE.get()
    if active is None or binding != active.binding:
        return None
    return copy.deepcopy(active.provenance)
