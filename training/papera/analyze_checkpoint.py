"""Read-only offline ACT checkpoint analysis; never trains or controls a robot.

Evaluate train and held-out windows with exactly the same eval-mode, z=0 prior.
Errors measure predicted accepted command targets, NOT measured robot error or
task success. All-window chunks overlap, so their targets are not independent
statistical observations. Stage groups refer to the observation anchor stage.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.papera.act_model import ACTConfig, ACTPolicy
from training.papera.data import STAGES, file_sha256, inside
from training.papera.train import (CHECKPOINT_VERSION, canonical_hash, json_write,
                                   move_batch, precision_context, seed_everything)

METRICS = ("normalized_action_l1", "xyz_error_mm", "gripper_error_mm", "wrapped_rpy_mae_degrees")
ARMS = ("left", "right")


def source_compatibility(provenance):
    """Use the evaluator's reviewed legacy tuples; never relax model identity."""
    from rollout.policy import verify_inference_sources
    return verify_inference_sources(provenance)[0]


def sequence_inputs(provenance):
    if provenance.get('sequence_data'):
        raise ValueError('Historical sequence overlays belong to the original project')
    return None, {}


def analysis_binding(provenance):
    """Independently recover the run-frozen formal binding, never CURRENT."""
    from training.papera.pack_data import VERSION, restore_binding
    if provenance.get("data_contract") != VERSION:
        return None
    frozen = provenance.get("pack_binding")
    if (not isinstance(frozen, dict) or frozen.get("pack_path") != provenance.get("pack")
            or frozen.get("manifest_sha256") != provenance.get("pack_manifest_sha256")
            or frozen.get("sequence_contract") != provenance.get("sequence_contract")
            or provenance.get("sequence_data") is not None):
        raise ValueError("Single-view analysis requires the exact frozen Safe Stitch binding")
    config = provenance["config"]
    if (Path(config["pack_path"]).resolve() != Path(frozen["pack_path"]).resolve()
            or config["image_size"] != frozen["image_size"]
            or provenance["model"]["num_cameras"] != 1
            or provenance["model"]["chunk_size"] != frozen["chunk_size"]):
        raise ValueError("Analysis model/preprocessing differs from the frozen Pack binding")
    return restore_binding(frozen)


def analysis_dataset(provenance, split, chunk_size, binding=None):
    """Reproduce the actual trained windows, including an explicit overfit subset."""
    overfit = bool(provenance.get("engineering_overfit"))
    keys = provenance.get("overfit_episode_keys")
    if overfit and (not isinstance(keys, list) or not 1 <= len(keys) <= 3
                    or len(set(keys)) != len(keys)):
        raise ValueError("Invalid overfit episode provenance")
    if provenance.get("data_contract") == "papera-safe-stitch-act-v1":
        from training.papera.pack_data import SafeStitchDataset
        binding = analysis_binding(provenance) if binding is None else binding
        if dict(binding) != provenance["pack_binding"]:
            raise ValueError("Analysis binding differs from training")
        expected_keys = keys if overfit else binding["split"]["train"]
        if binding["train_keys"] != expected_keys:
            raise ValueError("Analysis normalization/windows differ from the trained episode subset")
        return SafeStitchDataset(binding, split="train" if overfit else split,
                                 chunk_size=chunk_size, stride=1)
    raise ValueError('Analysis requires a current Safe Stitch run')


def command_errors(prediction, target, normalizer):
    """Return [window, horizon, arm, metric]; caller applies target validity."""
    if prediction.shape != target.shape or prediction.shape[-1] != 14:
        raise ValueError("Expected matching [..., 14] normalized actions")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise FloatingPointError("Nonfinite prediction or target")
    pred = normalizer.decode_action(prediction).astype(np.float64)
    true = normalizer.decode_action(target).astype(np.float64)
    errors = []
    for off in (0, 7):
        angle = (pred[..., off+3:off+6] - true[..., off+3:off+6] + np.pi) % (2*np.pi) - np.pi
        errors.append(np.stack((
            np.abs(prediction[..., off:off+7].astype(np.float64) - target[..., off:off+7]).mean(-1),
            np.linalg.norm(pred[..., off:off+3] - true[..., off:off+3], axis=-1) * 1000,
            np.abs(pred[..., off+6] - true[..., off+6]) * 1000,
            np.abs(angle).mean(-1) * 180/np.pi,
        ), axis=-1))
    return np.stack(errors, axis=-2)


def distribution(values):
    if not len(values):
        raise ValueError("Cannot summarize an empty group")
    return {"mean": float(np.mean(values)), "p50": float(np.quantile(values, .50)),
            "p90": float(np.quantile(values, .90)), "p95": float(np.quantile(values, .95)),
            "p99": float(np.quantile(values, .99)), "max": float(np.max(values))}


def grouped_metrics(errors, valid, episode_ids, stage_ids, entries, checkpoint, split, predictor):
    """Pooled errors weight valid command targets, not episodes or sessions."""
    rows = []
    n, horizon, _, _ = errors.shape
    sessions = np.asarray([entries[k]["session"] for k in episode_ids])
    groups = [("all", "all", np.ones(n, bool))]
    groups += [("episode", k, episode_ids == k) for k in sorted(set(episode_ids))]
    groups += [("session", k, sessions == k) for k in sorted(set(sessions))]
    groups += [("anchor_stage", STAGES[k], stage_ids == k) for k in sorted(set(stage_ids))]
    for group, value, windows in groups:
        # Report all chunk targets and first actionable target separately.
        for span, hsel in (("all", slice(None)), ("first", slice(0, 1))):
            selected, mask = errors[windows, hsel], valid[windows, hsel]
            if not mask.any():
                continue
            for arm, ai in (("both", slice(None)), ("left", slice(0, 1)), ("right", slice(1, 2))):
                pooled = selected[:, :, ai, :][mask].reshape(-1, len(METRICS))
                row = {"checkpoint": checkpoint, "split": split, "predictor": predictor,
                       "group": group, "value": str(value), "horizon": span, "arm": arm,
                       "windows": int(windows.sum()), "valid_targets_per_arm": int(mask.sum()),
                       "arm_target_pairs": len(pooled), "action_coordinates": len(pooled) * 7}
                for i, metric in enumerate(METRICS):
                    row.update({metric + "_" + stat: v for stat, v in distribution(pooled[:, i]).items()})
                rows.append(row)
    for hi in range(horizon):
        mask = valid[:, hi]
        if not mask.any():
            continue
        for arm, ai in (("both", slice(None)), ("left", slice(0, 1)), ("right", slice(1, 2))):
            pooled = errors[:, hi, ai, :][mask].reshape(-1, len(METRICS))
            row = {"checkpoint": checkpoint, "split": split, "predictor": predictor,
                   "group": "horizon", "value": str(hi), "horizon": str(hi), "arm": arm,
                   "windows": int(mask.sum()), "valid_targets_per_arm": int(mask.sum()),
                   "arm_target_pairs": len(pooled), "action_coordinates": len(pooled) * 7}
            for i, metric in enumerate(METRICS):
                row.update({metric + "_" + stat: v for stat, v in distribution(pooled[:, i]).items()})
            rows.append(row)
    return rows


def metric_self_test():
    """Hand-computed units, wrapped angles, asymmetric masks and weighting."""
    class Identity:
        def decode_action(self, x):
            return np.asarray(x).copy()
    target = np.zeros((2, 2, 14), np.float32)
    pred = target.copy()
    pred[0, 0, :3] = [.003, .004, 0]
    pred[0, 0, 6] = .002
    pred[0, 0, 3] = np.deg2rad(179)
    target[0, 0, 3] = np.deg2rad(-179)
    pred[0, 1, 7] = .010
    pred[1, :, :] = 99  # Must be excluded by the validity mask.
    errors = command_errors(pred, target, Identity())
    assert abs(errors[0, 0, 0, 1] - 5) < 1e-5
    assert abs(errors[0, 0, 0, 2] - 2) < 1e-5
    assert abs(errors[0, 0, 0, 3] - 2/3) < 1e-4
    valid = np.asarray([[True, True], [False, False]])
    rows = grouped_metrics(errors, valid, np.asarray(["a", "b"]), np.asarray([0, 1]),
                           {"a": {"session": "s"}, "b": {"session": "s"}}, "test", "val", "model")
    row = next(x for x in rows if x["group"] == "all" and x["arm"] == "both" and x["horizon"] == "all")
    assert row["action_coordinates"] == 28
    assert abs(row["xyz_error_mm_mean"] - 3.75) < 1e-5
    assert abs(row["normalized_action_l1_mean"] - np.abs(pred-target)[valid].mean()) < 1e-6
    return {"passed": True, "checks": ["3-4-5 mm Euclidean distance", "2 mm gripper width", "179/-179 degree wrap",
                                       "invalid targets excluded", "two arms and seven coordinates weighted"]}


def verify_inputs(run_path, run, binding=None):
    p = run["provenance"]
    if canonical_hash(p) != run["signature"]:
        raise ValueError("Run signature no longer matches provenance")
    source_compatibility(p)
    pack = Path(p["pack"])
    single = p.get("data_contract") == "papera-safe-stitch-act-v1"
    metadata_root = run_path if single else pack
    if single:
        binding = analysis_binding(p) if binding is None else binding
        if dict(binding) != p["pack_binding"]:
            raise ValueError("Frozen analysis binding was changed")
        for name, expected_value in (("stats.json", binding["stats"]), ("split.json", binding["split"]),
                                     ("pack_binding.json", dict(binding))):
            if json.loads((run_path/name).read_text(encoding="utf-8")) != expected_value:
                raise ValueError(f"Run-frozen metadata differs: {name}")
    expected = {pack/"manifest.json": p["pack_manifest_sha256"],
                metadata_root/"split.json": p["split_sha256"], metadata_root/"stats.json": p["stats_sha256"],
                ROOT/"training/papera/act_model.py": p["model_source_sha256"],
                # Historical readers/trainers are accepted only by the exact
                # reviewed tuple above; hash current files before AND after.
                ROOT/"training/papera/data.py": None,
                ROOT/"training/papera/train.py": None}
    if single:
        expected.update({run_path/"pack_binding.json": None,
            ROOT/"training/papera/pack_data.py": p["reader_source_sha256"],
            ROOT/"training/papera/data.py": p["normalizer_source_sha256"],
            ROOT/"dataset/task3/training_pack.py": p["pack_reader_source_sha256"]})
    if p.get("pretrained_weights_sha256"):
        expected[Path(p["model"]["backbone_weights"])] = p["pretrained_weights_sha256"]
    _, sequence_files = sequence_inputs(p)
    expected.update(sequence_files)
    manifest = json.loads((pack/"manifest.json").read_text(encoding="utf-8"))
    if single:
        for item in manifest["files"]:
            expected[inside(pack, pack/item["path"])] = item["sha256"]
    for entry in manifest["episodes"]:
        records = ((item["path"], item) for item in entry["files"]) if single else entry["files"].items()
        for rel, item in records:
            expected[inside(pack, pack/rel)] = item["sha256"]
    # Also fingerprint the immutable run records and both saved checkpoints.
    for rel in ("run.json", "completion.json", "metrics.jsonl", "checkpoints/best.pt", "checkpoints/last.pt"):
        expected[run_path/rel] = None
    result = {}
    for i, (path, expected_hash) in enumerate(expected.items()):
        got = file_sha256(path)
        if expected_hash is not None and expected_hash != got:
            raise ValueError(f"Input provenance mismatch: {path}")
        result[str(path)] = {"sha256": got, "size": path.stat().st_size}
        if (i+1) % 50 == 0:
            print(f"Input fingerprint check: {i+1}/{len(expected)}", flush=True)
    return result


@torch.no_grad()
def evaluate(checkpoint_path, run, split, out, include_mean_reference, binding=None):
    started = time.perf_counter()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if ckpt.get("version") != CHECKPOINT_VERSION or ckpt.get("signature") != run["signature"]:
        raise ValueError("Checkpoint version/signature mismatch")
    if ckpt.get("provenance") != run["provenance"]:
        raise ValueError("Checkpoint provenance mismatch")
    p, config = run["provenance"], run["provenance"]["config"]
    model_fields = dict(p["model"])
    # Entire trained state is restored; no reloading ImageNet initialization.
    model_fields["backbone_weights"] = None
    model = ACTPolicy(ACTConfig(**model_fields)).to("cuda")
    model.load_state_dict(ckpt["model"], strict=True)
    step = int(ckpt["step"])
    del ckpt
    model.eval()
    dataset = analysis_dataset(p, split, model.config.chunk_size, binding=binding)
    errors_list, reference_list, valid_list, prediction_list, target_list = [], [], [], [], []
    episode_list, stage_list, frame_list = [], [], []
    name = checkpoint_path.stem
    try:
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=True)
        for i, cpu in enumerate(loader):
            batch = move_batch(cpu, torch.device("cuda"))
            with precision_context(torch.device("cuda"), config["precision"]):
                output = model(batch["images"], batch["state"])
            pred = output["action"].float().cpu().numpy()
            target = cpu["action"].numpy()
            valid = ~cpu["is_pad"].numpy()
            errors_list.append(command_errors(pred, target, dataset.normalizer))
            if include_mean_reference:
                reference_list.append(command_errors(np.zeros_like(pred), target, dataset.normalizer))
            valid_list.append(valid)
            prediction_list.append(dataset.normalizer.decode_action(pred[:, 0]))
            target_list.append(dataset.normalizer.decode_action(target[:, 0]))
            episode_list.extend(cpu["episode_key"])
            stage_list.extend(cpu["stage"].numpy().tolist())
            frame_list.extend(cpu["raw_frame_index"].numpy().tolist())
            if (i+1) % 250 == 0 or i+1 == len(loader):
                print(f"{name} step={step} {split}: {min((i+1)*config['batch_size'],len(dataset))}/{len(dataset)} windows; {time.perf_counter()-started:.1f}s", flush=True)
        errors = np.concatenate(errors_list)
        valid = np.concatenate(valid_list)
        episode_ids = np.asarray(episode_list)
        stage_ids = np.asarray(stage_list)
        first_pred, first_target = np.concatenate(prediction_list), np.concatenate(target_list)
        rows = grouped_metrics(errors, valid, episode_ids, stage_ids, dataset.entries, name, split, "model")
        if include_mean_reference:
            ref_errors = np.concatenate(reference_list)
            rows += grouped_metrics(ref_errors, valid, episode_ids, stage_ids, dataset.entries, name, split, "train_mean")
        original_time = np.asarray([dataset.episodes[k]["original_time_s"][a] for k,a,_ in dataset.windows])
        single = p.get("data_contract") == "papera-safe-stitch-act-v1"
        segment_field = "train_sequence_id" if single else "segment_id"
        segments = np.asarray([dataset.episodes[k][segment_field][a] for k,a,_ in dataset.windows])
        epochs = np.asarray([dataset.episodes[k]["control_epoch"][a] for k,a,_ in dataset.windows])
        sessions = np.asarray([dataset.entries[k]["session"] for k in episode_ids])
        for i,(key, anchor, _) in enumerate(dataset.windows):
            if key != episode_ids[i] or dataset.episodes[key]["raw_frame_index"][anchor] != frame_list[i]:
                raise AssertionError("Prediction metadata order mismatch")
        virtual_metadata = ({"train_sequence_id": segments,
            "training_time_s": np.asarray([dataset.episodes[k]["training_time_s"][a] for k,a,_ in dataset.windows]),
            "action_control_tick": np.asarray([dataset.episodes[k]["action_control_tick"][a] for k,a,_ in dataset.windows])}
            if single else {})
        np.savez_compressed(out/f"{name}_{split}_first_targets.npz", prediction=first_pred, target=first_target,
                            valid=valid[:,0], episode_key=episode_ids, session=sessions,
                            raw_frame_index=np.asarray(frame_list), original_time_s=original_time,
                            segment_id=segments, control_epoch=epochs, anchor_stage_id=stage_ids,
                            first_target_errors=errors[:,0], metric_names=np.asarray(METRICS), **virtual_metadata)
        write_csv(out/f"{name}_{split}_metrics.csv", rows)
        result = {"checkpoint": name, "checkpoint_step": step, "split": split, "windows": len(dataset),
                  "episodes": len(set(episode_ids)), "sessions": len(set(sessions)),
                  "validation_scope": p.get("validation_scope", "heldout_sessions"),
                  "engineering_overfit": bool(p.get("engineering_overfit")),
                  "sequence_data": p.get("sequence_data"),
                  "sequence_contract": p.get("sequence_contract"),
                  "segment_field": segment_field,
                  "valid_action_targets": int(valid.sum()), "valid_action_coordinates": int(valid.sum())*14,
                  "valid_targets_per_horizon": valid.sum(0).astype(int).tolist(),
                  "elapsed_s": time.perf_counter()-started, "metrics": rows}
        json_write(out/f"{name}_{split}.json", result)
        print(json.dumps({"finished": f"{name}_{split}", "step": step, "pooled_all_chunk": rows[0]}, ensure_ascii=False), flush=True)
        return result
    finally:
        dataset.close()
        del model
        torch.cuda.empty_cache()


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    run_path = args.run.resolve()
    output = args.output.resolve() if args.output else run_path/"analysis/evaluation"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Analysis output must be new/empty; keep earlier analysis evidence")
    if output == run_path or output in run_path.parents:
        raise ValueError("Analysis output cannot be the run or its parent")
    checks = metric_self_test()
    run = json.loads((run_path/"run.json").read_text(encoding="utf-8"))
    config = run["provenance"]["config"]
    if config["validation_stride"] != 1:
        raise ValueError("This analyzer currently requires validation_stride=1 to match logged full validation")
    torch.set_num_threads(config["cpu_threads"])
    seed_everything(config["seed"])
    if not torch.cuda.is_available() or (config["precision"] == "bf16" and not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Original CUDA precision must be available")
    output.mkdir(parents=True, exist_ok=True)
    binding = analysis_binding(run["provenance"])
    before = verify_inputs(run_path, run, binding=binding)
    json_write(output/"inputs_before.json", before)
    json_write(output/"metric_self_test.json", checks)
    results = []
    for checkpoint, split, mean in (("best", "val", True), ("best", "train", True), ("last", "val", False)):
        results.append(evaluate(run_path/f"checkpoints/{checkpoint}.pt", run, split, output, mean, binding=binding))
    print("Checking all input hashes again after read-only evaluation", flush=True)
    after = verify_inputs(run_path, run, binding=binding)
    json_write(output/"inputs_after.json", after)
    if after != before:
        raise RuntimeError("Analysis inputs changed during evaluation")
    metrics_log = [json.loads(line) for line in (run_path/"metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    comparisons = []
    for r in results:
        if r["split"] != "val":
            continue
        log = next(x for x in metrics_log if x.get("event") == "validation" and x["step"] == r["checkpoint_step"])
        overall = next(x for x in r["metrics"] if x["group"]=="all" and x["arm"]=="both" and x["horizon"]=="all" and x["predictor"]=="model")
        comparisons.append({"checkpoint": r["checkpoint"], "step": r["checkpoint_step"],
                            "same_validation_scope": not log.get("smoke_subset", False),
                            "logged_prior_action_l1": log["prior_action_l1"],
                            "recomputed_prior_action_l1": overall["normalized_action_l1_mean"],
                            "absolute_difference": abs(log["prior_action_l1"]-overall["normalized_action_l1_mean"]),
                            "action_coordinates_match": r["valid_action_coordinates"] == log["valid_action_coordinates"]})
    if any(x["same_validation_scope"] and (x["absolute_difference"] > 2e-5 or not x["action_coordinates_match"]) for x in comparisons):
        raise ValueError("Recomputed validation does not reproduce logged metric within tolerance")
    summary = {"version": "papera-offline-analysis-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "run": str(run_path), "signature": run["signature"], "complete": True,
               "prior_only": True, "model_eval_mode": True, "trained_or_modified_originals": False,
               "all_input_hashes_unchanged": True, "source_sha256": file_sha256(Path(__file__)),
               "python": sys.version, "python_executable": sys.executable, "torch": str(torch.__version__),
               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "platform": platform.platform(),
               "precision": config["precision"], "batch_size": config["batch_size"],
               "cpu_threads": config["cpu_threads"], "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
               "original_run_git_commit": run["runtime"].get("git_commit"),
               "sequence_data": run["provenance"].get("sequence_data"),
               "sequence_contract": run["provenance"].get("sequence_contract"),
               "source_compatibility": source_compatibility(run["provenance"]),
               "engineering_overfit": bool(run["provenance"].get("engineering_overfit")),
               "validation_scope": run["provenance"].get("validation_scope", "heldout_sessions"),
               "smoke_log_comparison_note": "Smoke training validates only two batches; this analyzer evaluates the full split. Those scopes are reported, not asserted equal.",
               "elapsed_s": time.perf_counter()-started, "reproduction_checks": comparisons,
               "metric_self_test": checks, "results": [{k:v for k,v in r.items() if k!="metrics"} for r in results],
               "definitions": {"xyz_error_mm": "Euclidean error between predicted and demonstration accepted XYZ command; not actual end-effector or object error",
                   "wrapped_rpy_mae_degrees": "Mean absolute wrapped per-Euler-coordinate error; not an SO(3) geodesic angle",
                   "normalized_action_l1": "All 14 normalized coordinates; target-count weighting, including orientation coordinates",
                   "stage": "Category of observation anchor, not category of each future target",
                   "horizon": ("0 keeps its original next-control-tick target; future targets follow certified virtual 30Hz adjacency, not compressed physical time"
                               if binding else "0 is first next-control-tick target (~16.7 ms); following anchors ~33.3 ms apart, actual control discretization may differ"),
                   "train_mean": "Fixed zero in normalized action space, decoded to training-only unwrapped/circular-aware mean; no image/state input, not a deployment policy",
                   "distributions": "Valid target/arm samples, overlapping chunks included; quantiles are descriptive, not independent-example confidence intervals",
                   "first_targets": "One original next-tick prediction per eligible anchor. For Safe Stitch, segment_id aliases train_sequence_id; original and virtual clocks are both saved."}}
    json_write(output/"summary.json", summary)
    print(f"Read-only evaluation complete: {output}", flush=True)


if __name__ == "__main__":
    main()
