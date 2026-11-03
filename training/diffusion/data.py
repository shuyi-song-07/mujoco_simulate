"""Diffusion observations and future-only targets from the official Pack reader.

Safe Stitch adjacency and causal labels are consumed unchanged. The optional
file-hash audit is separate from the mandatory format/evidence validation.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from dataset.task3 import training_pack
from dataset.task3.prepare_paper_dataset import digest
from dataset.task3.sequence_stitching import VERSION as SEQUENCE_VERSION
from training.papera.data import ANGLE_COLUMNS, wrap_actions
from training.papera.pack_data import (
    PreparedPack, SafeStitchDataset, _close_episode, _metadata, resize_rgb,
)

VERSION = "papera-safe-stitch-diffusion-v1"
STATS_VERSION = "train-only-minmax-circular-rpy-v1"


def _data_config(config):
    result = {name: config.get(name, default) for name, default in (
        ("observation_horizon", 1), ("prediction_horizon", 30), ("action_horizon", 8))}
    result["image_size"] = list(config.get("image_size", (640, 480)))
    if (any(type(result[name]) is not int for name in (
            "observation_horizon", "prediction_horizon", "action_horizon"))
            or result["observation_horizon"] < 1 or result["prediction_horizon"] != 30
            or not 1 <= result["action_horizon"] <= result["prediction_horizon"]
            or len(result["image_size"]) != 2
            or any(type(value) is not int or value < 32 for value in result["image_size"])):
        raise ValueError("Expected positive observation horizon, future K=30, action horizon 1..30, image size >=32")
    return result


def _fit_statistics(episodes, training_keys):
    """Fit only selected training observations and valid causal action labels."""
    sine, cosine, action_count, state_count = np.zeros(14), np.zeros(14), 0, 0
    state_min, state_max = np.full(16, np.inf), np.full(16, -np.inf)
    for key in training_keys:
        arrays = episodes[key]
        state = np.asarray(arrays["state"], np.float64)
        action = np.asarray(arrays["action"], np.float64)[arrays["action_mask"].astype(bool)]
        if not len(state) or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("Cannot normalize missing/nonfinite training values")
        state_min = np.minimum(state_min, state.min(0))
        state_max = np.maximum(state_max, state.max(0))
        sine += np.sin(action).sum(0)
        cosine += np.cos(action).sum(0)
        state_count += len(state)
        action_count += len(action)
    if not action_count:
        raise ValueError("Cannot normalize missing training causal targets")
    reference = np.zeros(14, np.float64)
    reference[list(ANGLE_COLUMNS)] = np.arctan2(sine, cosine)[list(ANGLE_COLUMNS)]
    action_min, action_max = np.full(14, np.inf), np.full(14, -np.inf)
    for key in training_keys:
        arrays = episodes[key]
        action = wrap_actions(arrays["action"][arrays["action_mask"].astype(bool)], reference)
        if len(action):
            action_min = np.minimum(action_min, action.min(0))
            action_max = np.maximum(action_max, action.max(0))
    return {"version": STATS_VERSION, "fit_split": "train", "training_keys": list(training_keys),
        "state_count": state_count, "action_count": action_count,
        "state_min": state_min.tolist(), "state_max": state_max.tolist(),
        "action_min": action_min.tolist(), "action_max": action_max.tolist(),
        "action_angle_reference": reference.tolist(), "angle_columns": list(ANGLE_COLUMNS),
        "scale_floor": 1e-6, "output_range": [-1, 1], "clip": False}


class Normalizer:
    """Frozen train min/max transform; held-out values may exceed [-1, 1]."""

    def __init__(self, stats):
        if (stats.get("version") != STATS_VERSION or stats.get("fit_split") != "train"
                or stats.get("angle_columns") != list(ANGLE_COLUMNS)
                or stats.get("clip") is not False):
            raise ValueError("Incompatible frozen diffusion normalization")
        self.stats = copy.deepcopy(stats)
        self.reference = self._array("action_angle_reference", 14)
        floor = float(stats["scale_floor"])
        if not np.isfinite(floor) or floor <= 0:
            raise ValueError("Invalid normalization scale floor")
        for name, dim in (("state", 16), ("action", 14)):
            low, high = self._array(name+"_min", dim), self._array(name+"_max", dim)
            if np.any(high < low):
                raise ValueError("Invalid normalization bounds")
            center, scale = (low+high)/2, np.maximum((high-low)/2, floor)
            center.setflags(write=False)
            scale.setflags(write=False)
            setattr(self, name+"_center", center)
            setattr(self, name+"_scale", scale)

    def _array(self, name, dim):
        value = np.asarray(self.stats[name], dtype=np.float64)
        if value.shape != (dim,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid normalization field {name}")
        value.setflags(write=False)
        return value

    def encode_state(self, value):
        return ((np.asarray(value)-self.state_center)/self.state_scale).astype(np.float32)

    def decode_state(self, value):
        return (np.asarray(value)*self.state_scale+self.state_center).astype(np.float32)

    def encode_action(self, value):
        return ((wrap_actions(value, self.reference)-self.action_center)/self.action_scale).astype(np.float32)

    def decode_action(self, value):
        action = np.asarray(value)*self.action_scale+self.action_center
        action[..., ANGLE_COLUMNS] = (action[..., ANGLE_COLUMNS]+np.pi) % (2*np.pi)-np.pi
        return action.astype(np.float32)


def prepare_data(pack_path, task_id, config, check_hashes=False, engineering=False,
                 selected_train_keys=None):
    """Validate an explicit official Pack and freeze JSON-serializable metadata.

    No Raw or external Prepared data is read, and no alternate input is selected
    when the Pack is absent/incompatible. check_hashes=True opts into the older
    strict file audit and current task-source matching used by ACT.
    """
    root, config = Path(pack_path).resolve(), _data_config(config)
    checked = training_pack.verify(root, check_hashes=check_hashes)
    manifest = checked["manifest"]
    if manifest.get("task_id") != task_id:
        raise ValueError("Cross-task Pack input rejected")
    if bool(manifest.get("engineering_only")) != bool(engineering):
        raise ValueError("Engineering Pack requires explicit engineering scope; human and engineering data cannot mix")
    if (manifest.get("version") != training_pack.VERSION
            or manifest.get("sequence_contract") != SEQUENCE_VERSION
            or manifest.get("cameras") != ["oblique"]
            or manifest.get("state_dim") != 16 or manifest.get("action_dim") != 14
            or manifest.get("action_alignment") != training_pack.ALIGNMENT):
        raise ValueError("A complete current S16/A14 single-oblique Safe Stitch Pack is required")
    split = copy.deepcopy(manifest["split"])
    train_keys = list(split["train"] if selected_train_keys is None else selected_train_keys)
    if (not train_keys or len(set(train_keys)) != len(train_keys)
            or not set(train_keys) <= set(split["train"])):
        raise ValueError("Selected training keys must be unique members of the frozen train split")
    if selected_train_keys is not None and not engineering:
        raise ValueError("Selected training subsets require explicit engineering scope")
    metadata, training_arrays = {}, {}
    for entry in manifest["episodes"]:
        key = entry["key"]
        episode = training_pack.read_episode(root, key, checked=checked, include_evidence=False)
        try:
            metadata[key] = _metadata(episode, config["prediction_horizon"])
            if key in train_keys:
                training_arrays[key] = {name: episode["arrays"][name]
                    for name in ("state", "action", "action_mask")}
        finally:
            _close_episode(episode)
    stats = _fit_statistics(training_arrays, train_keys)
    counts = {}
    for side, keys in (("train", train_keys), ("val", split["val"])):
        counts[side] = {"episodes": len(keys)}
        for field in ("frames", "windows", "full_windows", "safe_stitch_windows", "invalid_action_anchors"):
            counts[side][field] = sum(metadata[key][field] for key in keys)
        counts[side]["safe_stitches"] = sum(len(metadata[key]["safe_stitch_positions"]) for key in keys)
        counts[side]["hard_boundaries"] = sum(len(metadata[key]["hard_boundary_positions"]) for key in keys)
    binding = PreparedPack(version=VERSION, pack_path=str(root), task_id=task_id,
        task_binding=copy.deepcopy(manifest["task_binding"]), engineering_only=bool(engineering),
        manifest_digest=digest(manifest), pack_version=manifest["version"],
        check_hashes=bool(check_hashes), verification=checked["verification"],
        sequence_contract=manifest["sequence_contract"], action_alignment=manifest["action_alignment"],
        cameras=["oblique"], state_dim=16, action_dim=14, config=config,
        chunk_size=config["prediction_horizon"], image_size=config["image_size"],
        cached_image_size=[manifest["config"]["image_width"], manifest["config"]["image_height"]],
        image_transform="whole_frame_PyAV_RGB24_reformat_then_CHW_float32_div255",
        target_semantics="future_only_original_next_control_tick_at_each_retained_anchor",
        history_semantics="past_and_current_within_certified_run_repeat_run_first_observation",
        split=split, train_keys=train_keys, validation_keys=list(split["val"]),
        stats=stats, episodes=metadata, counts=counts,
        frame_filter_applied=False, manual_exclusions_are_hard_boundaries=True)
    binding["binding_digest"] = digest(binding)
    binding.checked = checked
    return binding


def _validate_binding(binding):
    if digest({key: value for key, value in binding.items() if key != "binding_digest"}) != binding.get("binding_digest"):
        raise ValueError("Frozen diffusion data binding was modified")
    if (binding.get("version") != VERSION
            or _data_config(binding["config"]) != binding["config"]
            or binding["chunk_size"] != binding["config"]["prediction_horizon"]
            or binding["image_size"] != binding["config"]["image_size"]):
        raise ValueError("Incompatible diffusion data binding")


def restore_data(binding, check_hashes=False):
    """Reopen the frozen Pack without refitting or consulting current Raw data."""
    _validate_binding(binding)
    checked = training_pack.verify(binding["pack_path"], check_hashes=check_hashes)
    manifest = checked["manifest"]
    if (digest(manifest) != binding["manifest_digest"]
            or manifest["task_id"] != binding["task_id"]
            or manifest["task_binding"] != binding["task_binding"]
            or bool(manifest["engineering_only"]) != binding["engineering_only"]
            or manifest["split"] != binding["split"]):
        raise ValueError("Frozen diffusion Pack binding changed")
    if (not binding["train_keys"] or len(set(binding["train_keys"])) != len(binding["train_keys"])
            or not set(binding["train_keys"]) <= set(manifest["split"]["train"])
            or binding["validation_keys"] != manifest["split"]["val"]
            or binding["stats"]["training_keys"] != binding["train_keys"]):
        raise ValueError("Frozen diffusion training/validation selection changed")
    Normalizer(binding["stats"])
    for entry in manifest["episodes"]:
        episode = training_pack.read_episode(binding["pack_path"], entry["key"], checked=checked,
                                             include_evidence=False)
        try:
            if _metadata(episode, binding["chunk_size"]) != binding["episodes"][entry["key"]]:
                raise ValueError("Frozen diffusion sequence runs/windows changed")
        finally:
            _close_episode(episode)
    result = PreparedPack(copy.deepcopy(dict(binding)))
    result.checked = checked
    return result


class DiffusionDataset(SafeStitchDataset):
    """Official ACT future windows plus causal observation history.

    The inherited reader and pickle handling reopen RGB memmaps lazily in each
    Windows worker. Human auxiliary evidence is never an observation input.
    """

    def __init__(self, binding, split="train"):
        _validate_binding(binding)
        if not isinstance(binding, PreparedPack) or not hasattr(binding, "checked"):
            binding = restore_data(binding, check_hashes=binding["check_hashes"])
        if binding.get("version") != VERSION:
            raise ValueError("Incompatible diffusion data binding")
        self.observation_horizon = binding["config"]["observation_horizon"]
        super().__init__(binding, split=split, chunk_size=binding["chunk_size"],
                         normalizer=Normalizer(binding["stats"]))
        self._run_starts = {key: [start for start, _ in binding["episodes"][key]["runs"]]
                            for key in self.keys}

    def __getitem__(self, index):
        result = super().__getitem__(index)
        key, anchor, _ = self.windows[index]
        arrays = self.load_episode(key)
        starts = self._run_starts[key]
        run_start = starts[int(np.searchsorted(starts, anchor, side="right"))-1]
        history = np.maximum(np.arange(anchor-self.observation_horizon+1, anchor+1), run_start)
        result["images"] = np.stack([resize_rgb(self.images[key][0][position], self.image_size)
            .transpose(2, 0, 1)[None].astype(np.float32)/255. for position in history])
        result["state"] = self.normalizer.encode_state(arrays["state"][history])
        result["observation_raw_frame_index"] = arrays["raw_frame_index"][history].copy()
        result["observation_source_control_tick"] = arrays["source_control_tick"][history].copy()
        result["observation_train_sequence_id"] = arrays["train_sequence_id"][history].copy()
        return result
