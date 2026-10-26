"""Standard ACT samples from the official, verified Safe Stitch Training Pack.

No raw-data parsing, clipping, interpolation, or seam safety decision belongs
here. Dataset certificates define adjacency; training only chooses K targets
and masks the unavailable suffix. Original causal targets remain unchanged.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import av
import numpy as np

from dataset.task3 import training_pack
from dataset.task3.sequence_stitching import VERSION as SEQUENCE_VERSION
from training.papera.data import Normalizer, file_sha256, fit_statistics

VERSION = "papera-safe-stitch-act-v1"
CAMERAS = ("oblique",)


class PreparedPack(dict):
    """A serializable frozen binding plus a non-JSON, verified reader session."""


def resize_rgb(rgb, image_size=(320, 240)):
    """Whole-frame PyAV resize shared by training and closed-loop inference.

    image_size is (width, height). The raw cache stays untouched, uint8 RGB;
    conversion to CHW/float follows resizing at the caller.
    """
    image = np.asarray(rgb)
    width, height = map(int, image_size)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected one uint8 HWC RGB image")
    if min(width, height) < 32:
        raise ValueError("ACT input width and height must be at least 32")
    if image.shape[:2] == (height, width):
        return image
    return av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24").reformat(
        width=width, height=height, format="rgb24").to_ndarray()


def virtual_runs(arrays, sequence):
    """Consume certified sequence IDs and exclusions, never re-test physics.

    A manual exclusion is a hard training cut even if its physical seam was
    certified safe by Dataset. No retained observations are removed here.
    """
    n = len(arrays["state"])
    if not n:
        return [], [], []
    raw = arrays["raw_frame_index"]
    seq = arrays["train_sequence_id"]
    local = arrays["sequence_frame_index"]
    global_index = arrays["training_frame_index"]
    if not np.array_equal(global_index, np.arange(n)):
        raise ValueError("Pack virtual global index is not contiguous")
    if not np.allclose(arrays["training_time_s"], global_index / 30, atol=1e-10, rtol=0):
        raise ValueError("Pack virtual time differs from its certified 30 Hz clock")
    if not np.allclose(arrays["sequence_time_s"], local / 30, atol=1e-10, rtol=0):
        raise ValueError("Pack local virtual clock differs from the certificate")
    if not np.array_equal(arrays["action_mask"],
                          arrays["next_control_target_mask"] & arrays["causal_target_valid"]):
        raise ValueError("Causal action mask differs from Dataset contract")
    gaps = {int(gap["right_raw_frame_index"]): gap for gap in sequence["gaps"]}
    boundaries, safe_positions, hard_positions = [0], [], []
    for i in range(1, n):
        gap = gaps.get(int(raw[i]))
        epoch_changed = arrays["control_epoch"][i] != arrays["control_epoch"][i-1]
        sequence_changed = seq[i] != seq[i-1]
        if not sequence_changed and local[i] != local[i-1]+1:
            raise ValueError("Pack sequence-local index skips within a sequence")
        if sequence_changed and local[i] != 0:
            raise ValueError("Pack new sequence does not start at index zero")
        raw_gap = raw[i] != raw[i-1]+1
        if raw_gap and gap is None:
            raise ValueError("Removed-frame gap has no Dataset seam certificate")
        if gap is not None and (gap["left_raw_frame_index"] != int(raw[i-1])
                                or gap["kind"] not in ("safe_stitch", "hard_boundary")):
            raise ValueError("Dataset seam certificate endpoints/kind disagree")
        certificate_hard = gap is not None and gap["kind"] != "safe_stitch"
        if certificate_hard and not sequence_changed:
            raise ValueError("Unsafe seam shares a Dataset train_sequence_id")
        manual = gap is not None and gap["decision_origin"] == "manual"
        hard = sequence_changed or epoch_changed or certificate_hard or manual
        if hard:
            boundaries.append(i)
            hard_positions.append(i)
        elif gap is not None:
            safe_positions.append(i)
    return list(zip(boundaries, boundaries[1:]+[n])), safe_positions, hard_positions


def _metadata(episode, chunk_size):
    arrays = episode["arrays"]
    sequence = episode["diagnostics"]["virtual_sequence"]
    runs, safe, hard = virtual_runs(arrays, sequence)
    mask = arrays["action_mask"].astype(bool)
    valid_anchors = np.flatnonzero(mask).tolist()
    windows, full, crosses = 0, 0, 0
    for start, end in runs:
        for anchor in range(start, end):
            if not mask[anchor]:
                continue
            stop = min(end, anchor+chunk_size)
            windows += 1
            full += int(stop-anchor == chunk_size and mask[anchor:stop].all())
            crosses += int(any(anchor < position < stop for position in safe))
    return {"frames": len(mask), "runs": [list(run) for run in runs],
            "valid_anchors": valid_anchors, "safe_stitch_positions": safe,
            "hard_boundary_positions": hard, "windows": windows,
            "full_windows": full, "safe_stitch_windows": crosses,
            "invalid_action_anchors": int((~mask).sum())}


def _close_episode(episode):
    for image in episode.get("images", {}).values():
        handle = getattr(image, "_mmap", None)
        if handle is not None and not handle.closed:
            handle.close()


def prepare_pack(pack_path, selected_train_keys=None, chunk_size=30, *, checked=None,
                 image_size=(320, 240)):
    """Verify once and freeze paths, split, statistics, certificates and windows.

    Explicit engineering subsets must be a subset of the frozen training split.
    Their normalization is fitted only on those selected training episodes.
    The frozen validation membership is never shuffled or promoted into train.
    """
    root = Path(pack_path).resolve()
    if chunk_size != 30:
        raise ValueError("The current formal ACT contract is K=30")
    checked = training_pack.verify(root) if checked is None else checked
    if not isinstance(checked, training_pack.VerifiedPack):
        raise ValueError("checked must be the live official verify result")
    manifest = checked["manifest"]
    if (not checked.get("valid") or manifest.get("version") != training_pack.VERSION
            or manifest.get("sequence_contract") != SEQUENCE_VERSION
            or manifest.get("cameras") != list(CAMERAS)
            or manifest.get("state_dim") != 16 or manifest.get("action_dim") != 14
            or manifest.get("action_alignment") != training_pack.ALIGNMENT):
        raise ValueError("A complete current Safe Stitch single-oblique Pack is required")
    split = copy.deepcopy(manifest["split"])
    train_keys = list(split["train"] if selected_train_keys is None else selected_train_keys)
    if not train_keys or len(set(train_keys)) != len(train_keys) or not set(train_keys) <= set(split["train"]):
        raise ValueError("Selected training keys must be unique members of the frozen train split")
    if set(split["train"]) & set(split["val"]) or set(split["train_sessions"]) & set(split["val_sessions"]):
        raise ValueError("Frozen train/validation split overlaps")
    image_size = tuple(map(int, image_size))
    if len(image_size) != 2 or min(image_size) < 32:
        raise ValueError("Expected image_size=(width,height), each >=32")
    metadata, stats_arrays = {}, {}
    for entry in manifest["episodes"]:
        key = entry["key"]
        episode = training_pack.read_episode(root, key, checked=checked, include_evidence=False)
        try:
            metadata[key] = _metadata(episode, chunk_size)
            if key in train_keys:
                stats_arrays[key] = {field: episode["arrays"][field]
                                     for field in ("state", "action", "action_mask")}
        finally:
            _close_episode(episode)
    stats = fit_statistics(stats_arrays, train_keys)
    counts = {}
    for side, keys in (("train", train_keys), ("val", split["val"])):
        counts[side] = {"episodes": len(keys)}
        for field in ("frames", "windows", "full_windows", "safe_stitch_windows", "invalid_action_anchors"):
            counts[side][field] = sum(metadata[key][field] for key in keys)
        counts[side]["safe_stitches"] = sum(len(metadata[key]["safe_stitch_positions"]) for key in keys)
        counts[side]["hard_boundaries"] = sum(len(metadata[key]["hard_boundary_positions"]) for key in keys)
    binding = PreparedPack(version=VERSION, pack_path=str(root),
        manifest_sha256=file_sha256(root/"manifest.json"), pack_version=manifest["version"],
        sequence_contract=manifest["sequence_contract"], action_alignment=manifest["action_alignment"],
        cameras=list(CAMERAS), state_dim=16, action_dim=14, chunk_size=chunk_size,
        image_size=list(image_size), cached_image_size=[manifest["config"]["image_width"],
                                                       manifest["config"]["image_height"]],
        image_transform="whole_frame_PyAV_RGB24_reformat_then_CHW_float32_div255",
        split=split, train_keys=train_keys, validation_keys=list(split["val"]),
        stats=stats, episodes=metadata, counts=counts,
        frame_filter_applied=False, manual_exclusions_are_hard_boundaries=True)
    binding.checked = checked
    return binding


def restore_binding(binding):
    """Re-verify a JSON-frozen binding on resume, including stats and window IDs."""
    if binding.get("version") != VERSION:
        raise ValueError("Incompatible ACT data binding")
    result = prepare_pack(binding["pack_path"], selected_train_keys=binding["train_keys"],
                          chunk_size=binding["chunk_size"], image_size=binding["image_size"])
    if dict(result) != dict(binding):
        raise ValueError("Frozen ACT data binding changed: manifest/split/statistics/sequence")
    return result


class SafeStitchDataset:
    """Lazy official reads: workers reopen RGB mmaps instead of copying caches."""

    def __init__(self, binding, split="train", chunk_size=30, stride=1,
                 selected_keys=None, image_size=None):
        if split not in ("train", "val") or stride < 1 or chunk_size != binding["chunk_size"]:
            raise ValueError("Invalid split, stride, or frozen chunk length")
        self.binding = binding if isinstance(binding, PreparedPack) else restore_binding(binding)
        self.root = Path(binding["pack_path"])
        self.manifest = self.binding.checked["manifest"]
        self.entries = {entry["key"]: entry for entry in self.manifest["episodes"]}
        self.normalizer = Normalizer(binding["stats"])
        self.chunk_size = chunk_size
        self.image_size = tuple(binding["image_size"])
        if image_size is not None and tuple(image_size) != self.image_size:
            raise ValueError("Image shape differs from the frozen preprocessing contract")
        allowed = binding["train_keys"] if split == "train" else binding["validation_keys"]
        self.keys = list(allowed if selected_keys is None else selected_keys)
        if len(set(self.keys)) != len(self.keys) or not set(self.keys) <= set(allowed):
            raise ValueError("Dataset selection crosses the frozen split")
        self.episodes, self.images, self.windows = {}, {}, []
        self.sequence_manifest, self.sequence_arrays = None, {}
        self._pid = os.getpid()
        for key in self.keys:
            info = binding["episodes"][key]
            anchors = set(info["valid_anchors"])
            for start, end in info["runs"]:
                self.windows.extend((key, anchor, min(end, anchor+chunk_size))
                    for anchor in range(start, end, stride) if anchor in anchors)
        if not self.windows:
            raise ValueError(f"No valid action windows in {split}")

    def __len__(self):
        return len(self.windows)

    def load_episode(self, key):
        if key not in self.keys:
            raise ValueError("Episode is outside this dataset selection")
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if key not in self.episodes:
            episode = training_pack.read_episode(self.root, key, checked=self.binding.checked,
                                                  include_evidence=False)
            # Keep only arrays actually used by the standard loader/diagnostics.
            # Full raw human evidence remains in Pack and is never pickled.
            names = ("state", "action", "action_mask", "raw_frame_index", "source_control_tick",
                     "action_control_tick", "control_epoch", "original_time_s", "action_time_s",
                     "training_frame_index", "training_time_s", "train_sequence_id",
                     "sequence_frame_index", "sequence_time_s", "category_id", "category_mask",
                     "hand_normalized", "task_geometry_mask", "stage_id", "actual_done")
            self.episodes[key] = {name: episode["arrays"][name] for name in names}
            self.images[key] = [episode["images"]["oblique"]]
        return self.episodes[key]

    def close(self):
        for images in getattr(self, "images", {}).values():
            for image in images:
                mapping = getattr(image, "_mmap", None)
                if mapping is not None and not mapping.closed:
                    mapping.close()
        self.images, self.episodes = {}, {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["images"], state["episodes"], state["_pid"] = {}, {}, None
        return state

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __getitem__(self, index):
        key, anchor, end = self.windows[index]
        arrays = self.load_episode(key)
        length = end-anchor
        ids = np.minimum(np.arange(anchor, anchor+self.chunk_size), end-1)
        pad = (np.arange(self.chunk_size) >= length) | ~arrays["action_mask"][ids].astype(bool)
        action = self.normalizer.encode_action(arrays["action"][ids])
        action[pad] = 0
        image = resize_rgb(self.images[key][0][anchor], self.image_size)
        result = {
            "images": image.transpose(2, 0, 1)[None].astype(np.float32)/255.,
            "state": self.normalizer.encode_state(arrays["state"][anchor]),
            "action": action, "is_pad": pad,
            "gesture": arrays["category_id"][anchor].astype(np.int64),
            "gesture_mask": arrays["category_mask"][anchor].astype(bool),
            "skeleton": arrays["hand_normalized"][anchor].astype(np.float32),
            "skeleton_mask": arrays["task_geometry_mask"][anchor].astype(bool),
            "stage": np.int64(arrays["stage_id"][anchor]), "episode_key": key,
            "raw_frame_index": np.int64(arrays["raw_frame_index"][anchor]),
            "source_control_tick": np.int64(arrays["source_control_tick"][anchor]),
            "control_epoch": np.int64(arrays["control_epoch"][anchor]),
            "original_time_s": np.float64(arrays["original_time_s"][anchor]),
            "training_time_s": np.float64(arrays["training_time_s"][anchor]),
            "train_sequence_id": np.int64(arrays["train_sequence_id"][anchor]),
            "actual_done": bool(arrays["actual_done"][anchor]),
            "truncated": bool(length < self.chunk_size and not arrays["actual_done"][end-1]),
        }
        for output, field in (("target_raw_frame_index", "raw_frame_index"),
                              ("target_control_tick", "action_control_tick"),
                              ("target_original_time_s", "original_time_s"),
                              ("target_action_time_s", "action_time_s"),
                              ("target_training_time_s", "training_time_s"),
                              ("target_train_sequence_id", "train_sequence_id")):
            values = arrays[field][ids].copy()
            values[pad] = -1
            result[output] = values
        return result
