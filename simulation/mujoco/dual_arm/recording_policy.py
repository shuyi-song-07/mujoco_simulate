"""Shared recording/audit policy; deliberately independent of MuJoCo."""
from __future__ import annotations

import numpy as np

PHYSICAL_VELOCITY_NORM = 0.03
PHYSICAL_CHANGE_EPSILON = 1e-4
PHYSICAL_FIELDS = ("state", "action", "qpos")
VERIFY_STATES = frozenset({"GRASP_VERIFY", "RELEASE_VERIFY"})


def physical_change(snapshot, previous, task_state) -> bool:
    """Keep dynamics, verification, or change since the last retained frame.

    ``previous`` may be None or a subset containing state/action/qpos. Comparing
    with the last retained frame preserves slow cumulative settling, including
    during gesture confirmation or tracking pauses. The caller validates any
    archived array shapes and finite values before applying this policy.
    """
    dynamics = np.linalg.norm(np.asarray(snapshot.get("qvel", ()), dtype=float)) > PHYSICAL_VELOCITY_NORM
    changed = previous is not None and any(
        np.max(np.abs(np.asarray(snapshot[key]) - np.asarray(previous[key]))) > PHYSICAL_CHANGE_EPSILON
        for key in PHYSICAL_FIELDS if key in snapshot and key in previous
    )
    return bool(dynamics or changed or str(task_state) in VERIFY_STATES)
