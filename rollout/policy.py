"""Read-only inference adapter for authorized, exploratory MuJoCo evaluation.

The frozen training artifacts remain offline_only/deployable=False. This wrapper
does not turn pending human-quality reviews into approvals and is not a hardware
deployment interface. It returns unprotected robot targets; the rollout executor
must apply the same safety and timing rules to every experimental group.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import json
from pathlib import Path

import av
import numpy as np
import torch

from training.papera.act_model import ACTConfig, ACTPolicy
from training.papera.data import (
    ALIGNMENT, ANGLE_COLUMNS, CAMERAS, PACK_VERSION, Normalizer, file_sha256, inside,
)
from training.papera.train import CHECKPOINT_VERSION, canonical_hash, precision_context

ROOT = Path(__file__).resolve().parents[1]
# The physical state is arm-major, including each arm's actual finger width.
STATE_NAMES = tuple(name for side in ("left", "right")
                    for name in (*[f"{side}_joint{i}" for i in range(1, 8)], f"{side}_finger_width"))
ACTION_NAMES = tuple(f"{side}_{field}" for side in ("left", "right")
                     for field in ("x", "y", "z", "roll", "pitch", "yaw", "gripper_width_m"))
ACTION_UNITS = ("m", "m", "m", "rad", "rad", "rad", "m") * 2
SOURCE_PAIRS = (("act_model.py", "model_source_sha256"),
                ("data.py", "reader_source_sha256"), ("train.py", "trainer_source_sha256"))
class InvalidPolicyPrediction(ValueError):
    """A completed model forward returned unusable targets: a policy failure.

    Input/provenance errors and failed CUDA/model execution are different
    exceptions and must remain infrastructure errors in rollout accounting.
    """


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash_check(path, expected):
    got = file_sha256(path)
    if got != expected:
        raise ValueError(f"Provenance mismatch: {path}")
    return got


def verify_inference_sources(provenance):
    from training.papera.source_compatibility import verify_training_sources
    verify_training_sources(provenance)
    current=tuple(file_sha256(ROOT/'training/papera'/name) for name,_ in _source_pairs(provenance))
    return 'exact-current-training-sources',current,current


def _source_pairs(provenance):
    return (("act_model.py", "model_source_sha256"), ("pack_data.py", "reader_source_sha256"),
            ("train.py", "trainer_source_sha256")) if provenance.get("data_contract") == "papera-safe-stitch-act-v1" else SOURCE_PAIRS


def validate_statistics(stats, split):
    """Reject bad units/dimensions/scales before normalizing any observation."""
    if (stats.get("version") != "train-only-zscore-circular-rpy-v1"
            or stats.get("fit_split") != "train"
            or stats.get("training_keys") != split["train"]
            or stats.get("angle_columns") != list(ANGLE_COLUMNS)):
        raise ValueError("Normalization contract or training split mismatch")
    for key, length in (("state_mean", 16), ("state_std", 16), ("action_mean", 14),
                        ("action_std", 14), ("action_angle_reference", 14)):
        values = np.asarray(stats[key], dtype=np.float64)
        if values.shape != (length,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid normalization array: {key}")
        if key.endswith("std") and np.any(values <= 0):
            raise ValueError(f"Normalization scale must be positive: {key}")


def _verify_single_pack(run_dir, provenance, verify_cache):
    """Consume the formal pack and the run-frozen ACT adapter binding."""
    from dataset.task3 import training_pack
    from training.papera import pack_data
    pack = Path(provenance["pack"]).resolve()
    binding = provenance.get("pack_binding", {})
    config = provenance["config"]
    if (provenance.get("data_contract") != pack_data.VERSION
            or provenance.get("action_alignment") != training_pack.ALIGNMENT
            or binding.get("version") != pack_data.VERSION
            or binding.get("pack_path") != str(pack)
            or binding.get("manifest_sha256") != provenance["pack_manifest_sha256"]
            or binding.get("sequence_contract") != provenance.get("sequence_contract")
            or binding.get("cameras") != ["oblique"]
            or provenance.get("cameras") != ["oblique"]
            or binding.get("chunk_size") != 30
            or provenance["model"].get("num_cameras") != 1
            or Path(config.get("pack_path", "")).resolve() != pack):
        raise ValueError("Incompatible frozen single-view Safe Stitch ACT binding")
    hashes = {}
    for path, expected in ((pack/"manifest.json", provenance["pack_manifest_sha256"]),
                           (run_dir/"stats.json", provenance["stats_sha256"]),
                           (run_dir/"split.json", provenance["split_sha256"]),
                           (ROOT/"dataset/task3/training_pack.py", provenance["pack_reader_source_sha256"]),
                           (ROOT/"training/papera/data.py", provenance["normalizer_source_sha256"])):
        hashes[str(path)] = _hash_check(path, expected)
    manifest = _read(pack/"manifest.json")
    split, stats = _read(run_dir/"split.json"), _read(run_dir/"stats.json")
    if (manifest.get("version") != training_pack.VERSION or manifest.get("complete") is not True
            or manifest.get("sequence_contract") != binding["sequence_contract"]
            or manifest.get("cameras") != ["oblique"] or manifest.get("split") != split
            or split != binding.get("split") or stats != binding.get("stats")
            or manifest.get("action_alignment") != training_pack.ALIGNMENT
            or binding.get("image_size") != config.get("image_size")
            or binding.get("cached_image_size") != [manifest["config"]["image_width"],
                                                     manifest["config"]["image_height"]]):
        raise ValueError("Single-view manifest/statistics/split differs from frozen training")
    statistics_split = dict(split, train=binding["train_keys"])
    validate_statistics(stats, statistics_split)
    if verify_cache:
        checked = training_pack.verify(pack)
        rebuilt = pack_data.prepare_pack(pack, selected_train_keys=binding["train_keys"],
            chunk_size=30, checked=checked, image_size=binding["image_size"])
        if dict(rebuilt) != binding:
            raise ValueError("Frozen ACT binding differs from the independently verified Pack")
    from rollout.protocol import load_scene_provenance
    scene = load_scene_provenance(pack)
    camera_configs = {canonical_hash(value["camera"]): value["camera"]
                      for value in scene["oblique_render_sources"].values()}
    count = sum(len(entry["files"]) for entry in manifest["episodes"]) if verify_cache else 0
    return pack, manifest, split, stats, hashes, camera_configs, count


def _verify(run_dir, checkpoint, verify_cache, allow_overfit=False):
    run_dir = Path(run_dir).resolve()
    if checkpoint not in ("best", "last"):
        raise ValueError("checkpoint must be 'best' or 'last'; no arbitrary unverified checkpoint")
    run = _read(run_dir / "run.json")
    provenance = run["provenance"]
    if canonical_hash(provenance) != run["signature"]:
        raise ValueError("Run signature mismatch")
    config = provenance["config"]
    single_view = provenance.get("data_contract") == "papera-safe-stitch-act-v1"
    if not single_view: raise ValueError('Use the original project for historical three-view checkpoints')
    completion = _read(run_dir / "completion.json")
    overfit_flags = [bool(record.get("engineering_overfit", False))
                    for record in (provenance, run["runtime"], completion)]
    if len(set(overfit_flags)) != 1:
        raise ValueError("Overfit diagnostic scope differs between run and completion")
    overfit = overfit_flags[0]
    if overfit and not allow_overfit:
        raise ValueError("Overfit diagnostic models require explicit engineering evaluation, not a formal batch")
    if ((not allow_overfit and (run["runtime"].get("engineering_smoke") is not False
            or completion.get("engineering_smoke") is not False))
            or completion.get("completed") is not True
            or completion["step"] != completion["target_steps"]
            or completion["step"] != config["steps"] or completion["step"] <= 0):
        raise ValueError("Evaluation requires a complete non-engineering training run")
    expected_alignment = "next_control_tick_at_raw_observation_anchors" if single_view else ALIGNMENT
    if provenance.get("action_alignment") != expected_alignment:
        raise ValueError("Checkpoint action timing is incompatible")
    if run["runtime"]["precision"] != config["precision"]:
        raise ValueError("Recorded precision mismatch")
    if str(torch.__version__) != run["runtime"]["torch"]:
        raise ValueError("Use the original paperA_train torch version for this comparison")
    model_fields = dict(provenance["model"])
    model_config = ACTConfig(**model_fields)
    expected_model = dict(config["model"], variant=provenance["variant"])
    if expected_model.get("backbone_weights") is not None:
        expected_model["backbone_weights"] = str((ROOT / expected_model["backbone_weights"]).resolve())
    # Match train.py's ACTConfig expansion: optional defaults need not appear in
    # the input JSON, but are always recorded in provenance.model. Source hashes
    # below still pin the implementation that supplies those defaults.
    expected_model = asdict(ACTConfig(**expected_model))
    if model_fields != expected_model:
        raise ValueError("Run model differs from frozen configuration")
    pack = Path(provenance["pack"]).resolve()
    hashes = {}
    if single_view:
        pack, manifest, split, stats, hashes, camera_configs, cache_count = _verify_single_pack(
            run_dir, provenance, verify_cache)
    else:
        for relative, key in (("manifest.json", "pack_manifest_sha256"),
                              ("stats.json", "stats_sha256"), ("split.json", "split_sha256")):
            hashes[str(pack / relative)] = _hash_check(pack / relative, provenance[key])
    source_compatibility, current_sources, historical_sources = verify_inference_sources(provenance)
    for (name, _), value in zip(_source_pairs(provenance), current_sources):
        hashes[str(ROOT / "training/papera" / name)] = value
    if provenance.get("pretrained_weights_sha256"):
        path = Path(model_fields["backbone_weights"])
        hashes[str(path)] = _hash_check(path, provenance["pretrained_weights_sha256"])
    if not single_view:
        manifest, split, stats = (_read(pack / name) for name in ("manifest.json", "split.json", "stats.json"))
    if not single_view and (manifest.get("version") != PACK_VERSION or manifest.get("complete") is not True
            or manifest.get("action_alignment") != ALIGNMENT
            or manifest.get("cameras") != list(CAMERAS)
            or manifest["config"]["image_width"] != 320
            or manifest["config"]["image_height"] != 240
            or manifest["config"]["image_order"] != list(CAMERAS)):
        raise ValueError("Unsupported or incomplete training image/data contract")
    for name in (() if single_view else ("stats.json", "split.json")):
        if manifest["files"][name]["sha256"] != hashes[str(pack / name)]:
            raise ValueError(f"Manifest metadata hash mismatch: {name}")
    if (set(split["train"]) & set(split["val"])
            or set(split["train_sessions"]) & set(split["val_sessions"])):
        raise ValueError("Training/validation split leakage")
    if not single_view:
        validate_statistics(stats, split)
    sequence_metadata = None
    if provenance.get('sequence_data') or config.get('sequence_data'):
        raise ValueError('Only the official Safe Stitch Pack contract is supported')
    overfit_keys = provenance.get("overfit_episode_keys")
    if overfit and (not isinstance(overfit_keys, list)
                    or not 1 <= len(overfit_keys) <= 3
                    or len(set(overfit_keys)) != len(overfit_keys)
                    or not set(overfit_keys).issubset(split["train"])
                    or config["steps"] > 3000):
        raise ValueError("Overfit diagnostic has an invalid training subset or update budget")
    if single_view and provenance["pack_binding"]["train_keys"] != (overfit_keys if overfit else split["train"]):
        raise ValueError("Normalization/training membership differs from the declared run scope")
    if not manifest["episodes"]:
        raise ValueError("Empty training manifest")
    if not single_view:
        camera_configs, cache_count = {}, 0
    for entry in ([] if single_view else manifest["episodes"]):
        if (entry.get("state_names") != list(STATE_NAMES)
                or entry.get("action_names") != list(ACTION_NAMES)
                or entry.get("action_units") != list(ACTION_UNITS)):
            raise ValueError("Training state/action order or units mismatch")
        camera_configs[canonical_hash(entry["cameras"])] = entry["cameras"]
        if verify_cache:
            for relative, record in entry["files"].items():
                path = inside(pack, pack / relative)
                if path.stat().st_size != record["size"]:
                    raise ValueError(f"Cache size mismatch: {path}")
                hashes[str(path)] = _hash_check(path, record["sha256"])
                cache_count += 1
    ckpt_path = run_dir / "checkpoints" / f"{checkpoint}.pt"
    before_hash = file_sha256(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if file_sha256(ckpt_path) != before_hash:
        raise ValueError("Checkpoint changed while loading")
    if (ckpt.get("version") != CHECKPOINT_VERSION or ckpt.get("signature") != run["signature"]
            or ckpt.get("provenance") != provenance):
        raise ValueError("Checkpoint provenance/version mismatch")
    validations = [row for row in (json.loads(line) for line in
                   (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
                   if row.get("event") == "validation"]
    if not validations or any(not np.isfinite(row["prior_action_l1"]) for row in validations):
        raise ValueError("Missing/nonfinite validation selection evidence")
    best = min(validations, key=lambda row: row["prior_action_l1"])
    expected_step = best["step"] if checkpoint == "best" else completion["step"]
    if ckpt["step"] != expected_step:
        raise ValueError("Selected checkpoint step does not match the frozen selection rule")
    if not np.isclose(completion["best_prior_action_l1"], best["prior_action_l1"], rtol=0, atol=1e-12):
        raise ValueError("Completion best metric differs from validation history")
    if not ckpt.get("model") or any(not torch.isfinite(v).all().item()
            for v in ckpt["model"].values() if torch.is_tensor(v) and (v.is_floating_point() or v.is_complex())):
        raise ValueError("Checkpoint has missing or nonfinite model weights")
    metadata = {
        "version": "papera-exploratory-inference-v1", "run_dir": str(run_dir),
        "run_signature": run["signature"], "variant": provenance["variant"],
        "checkpoint": str(ckpt_path), "checkpoint_sha256": before_hash,
        "checkpoint_step": int(ckpt["step"]), "selected_by": "validation/prior_action_l1",
        "pack": str(pack), "checked_hashes": hashes, "verified_cache_files": cache_count,
        "full_cache_verified": bool(verify_cache), "training_precision": config["precision"],
        "cameras": ["oblique"] if single_view else list(CAMERAS), "training_camera_configs": camera_configs,
        "state_names": list(STATE_NAMES), "action_names": list(ACTION_NAMES),
        "action_units": list(ACTION_UNITS), "action_alignment": expected_alignment,
        "chunk_size": model_config.chunk_size, "source_image_shape": [480, 640, 3],
        "model_image_shape": ([1, 3, *reversed(provenance["pack_binding"]["image_size"])]
                              if single_view else [3, 3, 240, 320]),
        "image_preprocess": (provenance["pack_binding"]["image_transform"] if single_view else
                             "PyAV RGB24 reformat to 320x240; CHW float32 / 255; ImageNet normalization is internal"),
        "image_domain_note": "Live RGB rendering avoids the lossy video/YUV decode used by the recorded training cache; no artificial compression is inserted.",
        "prior_z_zero": True, "human_inference_input": False,
        "engineering_overfit": overfit,
        "overfit_episode_keys": overfit_keys,
        "sequence_data": provenance.get("sequence_data"),
        "data_contract": provenance.get("data_contract"),
        "sequence_contract": provenance.get("sequence_contract"),
        "source_compatibility": source_compatibility,
        "historical_training_sources": dict(zip((key for _, key in SOURCE_PAIRS), historical_sources)),
        "exploratory": True, "human_quality_review": provenance.get("human_quality_review"),
        "original_deployable": completion.get("deployable"),
        "original_offline_only": provenance.get("offline_only"),
        "scope": "user-authorized standalone MuJoCo evaluation; not hardware deployment",
    }
    return metadata, ckpt, stats


def verify_run_provenance(run_dir, checkpoint="best", *, verify_cache=True, allow_overfit=False):
    """Parent performs full cache verification once; worker loaders can omit it."""
    metadata, _, _ = _verify(run_dir, checkpoint, verify_cache, allow_overfit)
    return metadata


def prepare_observation(images, state, normalizer, *, cameras=CAMERAS, image_size=(320, 240)):
    """Full-frame RGB -> the existing cache's PyAV resize, no crop or flip."""
    cameras = tuple(cameras)
    if cameras not in (tuple(CAMERAS), ("oblique",)):
        raise ValueError("Unsupported policy camera contract")
    if not isinstance(images, Mapping) or set(images) != set(cameras):
        raise ValueError(f"Exactly {cameras} RGB camera observations are required")
    state = np.asarray(state)
    if (state.shape != (16,) or not np.issubdtype(state.dtype, np.number)
            or np.iscomplexobj(state) or not np.isfinite(state).all()):
        raise ValueError("Robot state must be 16 finite joint-angle/actual-width values")
    prepared = []
    for name in cameras:
        image = np.asarray(images[name])
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError(f"{name}: expected RGB uint8 [480,640,3], without crop/flip")
        if cameras == ("oblique",):
            from training.papera.pack_data import resize_rgb
            resized = resize_rgb(image, image_size=image_size)
        else:
            resized = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24").reformat(
                width=320, height=240, format="rgb24").to_ndarray()
        prepared.append(resized.transpose(2, 0, 1))
    image_array = np.stack(prepared).astype(np.float32) / 255.
    normalized_state = normalizer.encode_state(state)
    if not np.isfinite(normalized_state).all():
        raise ValueError("Normalized robot state is nonfinite")
    return image_array, normalized_state


class Policy:
    """Frozen eval-mode ACT; predicts one raw 30x14 target chunk per observation."""
    def __init__(self, run_dir, checkpoint="best", device="cuda", *, verify_cache=False, allow_overfit=False):
        self.metadata, ckpt, stats = _verify(run_dir, checkpoint, verify_cache, allow_overfit)
        self.device = torch.device(device)
        if self.device.type not in ("cuda", "cpu"):
            raise ValueError("Only explicit CUDA evaluation or CPU engineering inference is supported")
        self.precision = self.metadata["training_precision"]
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; no implicit CPU fallback for evaluation")
            if self.precision == "bf16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("Original BF16 inference is unavailable")
        fields = dict(ckpt["provenance"]["model"])
        fields["backbone_weights"] = None  # Entire trained state is loaded, no reinitialization/download.
        self.model = ACTPolicy(ACTConfig(**fields))
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.requires_grad_(False).eval().to(self.device)
        self.normalizer = Normalizer(stats)
        self.metadata.update(device=str(self.device), torch=str(torch.__version__),
                             inference_precision=self.precision if self.device.type == "cuda" else "fp32",
                             training_precision_matched=self.device.type == "cuda" or self.precision == "fp32",
                             model_eval=True, posterior_inputs=False, gradients=False)

    @torch.inference_mode()
    def predict(self, images, state):
        height, width = self.metadata["model_image_shape"][-2:]
        image_array, normalized_state = prepare_observation(images, state, self.normalizer,
            cameras=self.metadata["cameras"], image_size=(width, height))
        return self._predict_prepared(image_array, normalized_state)

    @torch.inference_mode()
    def _predict_prepared(self, image_array, normalized_state):
        """Internal path also permits exact cached-observation engineering checks."""
        image_array = np.asarray(image_array)
        normalized_state = np.asarray(normalized_state)
        if (image_array.shape != tuple(self.metadata["model_image_shape"]) or normalized_state.shape != (16,)
                or not np.isfinite(image_array).all() or not np.isfinite(normalized_state).all()
                or image_array.min() < 0 or image_array.max() > 1):
            raise ValueError("Invalid prepared observation")
        images = torch.as_tensor(image_array, dtype=torch.float32, device=self.device).unsqueeze(0)
        state = torch.as_tensor(normalized_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with precision_context(self.device, self.precision):
            output = self.model(images, state)  # No hand labels, no future actions, z=0.
        pred = output["action"].float().cpu().numpy()[0]
        if pred.shape != (self.model.config.chunk_size, 14) or not np.isfinite(pred).all():
            raise InvalidPolicyPrediction("Model prediction has invalid shape/nonfinite values")
        action = self.normalizer.decode_action(pred)
        if not np.isfinite(action).all():
            raise InvalidPolicyPrediction("Decoded robot action is nonfinite")
        # Do not clip widths/poses here: the common executor owns limits and logs interventions.
        return action
