"""Describe the configuration and implementation loaded by a local service.

Identity is captured at process startup, never recomputed by /health. A launcher
can therefore reject a process that predates edits without stopping that process.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import PROJECT_ROOT


def runtime_fingerprint(root=PROJECT_ROOT):
    root = Path(root)
    paths = sorted(p for p in (root / "simulation").rglob("*")
                   if p.is_file() and p.suffix.lower() in {".py", ".xml", ".obj", ".stl"})
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def backend_identity(config, *, input_kind, seed, headless, previews, root=PROJECT_ROOT):
    return {
        "version": "task3-service-v1",
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "runtime_sha256": runtime_fingerprint(root),
        "input_kind": input_kind,
        "session_seed": seed,
        "headless": bool(headless),
        "previews": bool(previews),
    }
