"""Fingerprint the physical rendering definition independently of caller wiring."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from common.layout import source_path, logical_source_path


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def render_identity(config: dict, model=None) -> dict:
    """Return actual FOV and complete scene/mesh/source hashes.

    Callers may supply their already-compiled model. Hashes deliberately do not
    include recorder code or the complete render-worker module: adding metadata
    plumbing cannot change a camera. Every invocation rereads source bytes, so
    a same-size asset rewrite cannot evade verification through a stat cache.
    """
    import mujoco
    from simulation.mujoco.dual_arm.config import DEFAULT_SCENE_PATH
    from tasks.runtime import create_model
    from .render_workers import apply_snapshot
    from .training_camera import training_camera_metadata

    model = create_model(config) if model is None else model
    fovy = float(model.vis.global_.fovy)
    if not math.isfinite(fovy) or not 0 < fovy < 180:
        raise ValueError("Model vertical field of view must be finite and in (0,180)")
    project = Path(__file__).resolve().parents[3]
    binding = config.get('task_binding')
    scene_path = (project / binding['model_xml']).resolve() if binding else Path(DEFAULT_SCENE_PATH).resolve()
    files = [scene_path,
             project / "simulation/mujoco/dual_arm/task_controller.py",
             project / "simulation/mujoco/dual_arm/task_scenes.py",
             project / "simulation/mujoco/dual_arm/training_camera.py",
             project / "simulation/mujoco/dual_arm/config.py"]
    if binding:
        files.extend(project / name for name in binding['sources'] if Path(name).suffix in ('.py', '.json'))
    source_files = {logical_source_path(source_path(path)).relative_to(project).as_posix(): _sha(source_path(path)) for path in files}
    assets = project / binding['mesh_directory'] if binding else scene_path.parent.parent / "franka_emika_panda/assets"
    mesh_hashes = {}
    for node in ET.parse(scene_path).getroot().findall("asset/mesh"):
        name = node.get("file")
        path = (assets / name).resolve()
        if not path.is_relative_to(assets.resolve()):
            raise ValueError(f"Mesh path escapes renderer assets: {name}")
        mesh_hashes[path.relative_to(project).as_posix()] = _sha(path)
    payload = {
        "version": "task3-render-identity-v1",
        "config_hash": _digest(config),
        "camera": training_camera_metadata(config),
        "actual_fovy_degrees": fovy,
        "model_dimensions": {name: int(getattr(model, name)) for name in ("nq", "nv", "nu", "nbody")},
        "mujoco_version": mujoco.__version__,
        "source_files_sha256": source_files,
        "mesh_files_sha256": mesh_hashes,
        "snapshot_restore_function_sha256": hashlib.sha256(inspect.getsource(apply_snapshot).encode()).hexdigest(),
        "appearance": "opaque model geometry; no operator-view overrides",
    }
    return {**payload, "identity_sha256": _digest(payload)}
