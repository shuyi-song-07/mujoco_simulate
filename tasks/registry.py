"""Task discovery, versioned bindings and explicitly registered runtime hooks."""
from __future__ import annotations
import copy
import hashlib
import importlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = {"state": "dual_panda_q7_width_m_s16", "action": "dual_xyz_rpy_width_m_a14",
          "observation": "oblique_rgb", "state_dim": 16, "action_dim": 14}
HOOKS = ("initial_state", "configure_xml", "reset", "collection_success", "rollout_monitor", "preflight", "dataset_adapter")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def local(path):
    p = (ROOT / path).resolve()
    if not p.is_relative_to(ROOT) or not p.is_file():
        raise ValueError(f"Missing or external project resource: {path}")
    return p

def entrypoint(value):
    module, separator, name = value.partition(":")
    if not separator or not module.startswith("tasks."):
        raise ValueError(f"Task entrypoint must be tasks.module:function: {value}")
    result = getattr(importlib.import_module(module), name)
    if not callable(result):
        raise ValueError(f"Not callable: {value}")
    return result

class Task:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        m = self.manifest
        required = {"task_id", "display_name", "status", "version", "scene_id", "scene_config",
                    "model_xml", "mesh_directory", "controller", "schema", "hooks", "source_files"}
        if required - m.keys():
            raise ValueError(f"{self.path}: missing fields {sorted(required - m.keys())}")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", m["task_id"]):
            raise ValueError(f"Invalid task_id: {m['task_id']}")
        if self.path.parent.name != m["task_id"] or m["status"] not in ("ready", "development"):
            raise ValueError(f"Task directory/status mismatch: {self.path}")
        self.id = m["task_id"]
        if m["schema"] != SCHEMA:
            raise ValueError(f"{self.id}: unsupported schema; implement and register an explicit adapter first")
        if m["controller"] != "dual_panda_v1":
            raise ValueError(f"{self.id}: unsupported controller")
        if m["status"] == "ready":
            if not m.get('evaluation_config'):
                raise ValueError(f'{self.id}: missing evaluation_config')
            local(m['evaluation_config'])
            if set(m["hooks"]) != set(HOOKS):
                raise ValueError(f"{self.id}: hooks must be {HOOKS}")
            for name in (m["scene_config"], m["model_xml"], *m["source_files"]):
                local(name)
            mesh = (ROOT / m["mesh_directory"]).resolve()
            if not mesh.is_relative_to(ROOT) or not mesh.is_dir():
                raise ValueError(f"Missing mesh directory: {mesh}")
            for hook in m["hooks"].values():
                entrypoint(hook)

    def require_ready(self):
        if self.manifest["status"] != "ready":
            raise ValueError(f"{self.id} is development: {self.manifest.get('reason', 'not implemented')}")
        return self

    def binding(self):
        self.require_ready()
        m = self.manifest
        files = {self.path.relative_to(ROOT).as_posix(), m["scene_config"], m["model_xml"],
                 "config/collection.json", "config/simulation_base.json", *m["source_files"]}
        files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/m["mesh_directory"]).iterdir() if p.is_file())
        return {"version": "papera-task-binding-v1", "task_id": self.id, "task_version": m["version"],
                "scene_id": m["scene_id"], "manifest_sha256": sha(self.path), "schema": copy.deepcopy(m["schema"]),
                "hooks": copy.deepcopy(m["hooks"]), "model_xml": m["model_xml"],
                "mesh_directory": m["mesh_directory"], "sources": {f: sha(local(f)) for f in sorted(files)}}

    def config(self):
        from simulation.mujoco.dual_arm.config import load_config
        config = load_config(ROOT/"config/collection.json")
        patch = json.loads(local(self.manifest["scene_config"]).read_text(encoding="utf-8"))
        config["scene"].update(patch["scene"])
        config["scene"]["scenario"] = self.manifest["scene_id"]
        config["task_binding"] = self.binding()
        return config

def discover(directory=None):
    result = {}
    for path in sorted(Path(directory or ROOT/"tasks").glob("*/task.json")):
        task = Task(path)
        if task.id in result:
            raise ValueError(f"Duplicate task: {task.id}")
        result[task.id] = task
    return result

def get(task_id):
    tasks = discover()
    if task_id not in tasks:
        raise ValueError(f"Unknown task {task_id!r}; available: {', '.join(tasks)}")
    return tasks[task_id].require_ready()

def validate_binding(binding, task_id=None):
    if not isinstance(binding, dict) or not binding.get("task_id"):
        raise ValueError("Missing Task Registry binding; historical data cannot silently enter this pipeline")
    if task_id is not None and binding["task_id"] != task_id:
        raise ValueError(f"Cross-task input rejected: {binding['task_id']} != {task_id}")
    task = get(binding["task_id"])
    if binding != task.binding():
        raise ValueError(f"Frozen task/assets differ from current task {task.id}; restore exact task sources or register a new version")
    return task

def hook(config, name):
    binding = config.get("task_binding")
    if not binding:
        raise ValueError("Registered task_binding required")
    return entrypoint(binding["hooks"][name])
