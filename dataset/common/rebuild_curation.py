"""Rebuild the one curation policy from raw evidence without re-encoding video."""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from dataset.task3.curation_io import load_episode
from dataset.task3.curation_rules import derive_episode, load_config, validate_config, effective_config
from dataset.task3.curation_config import excluded_episode_reason
from dataset.task3.prepare_paper_dataset import (
    check_fingerprints, digest, inside, write_json, write_rows, write_analysis_artifacts,
    implementation_fingerprints,
)

POLICY = "papera-intent-aware"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_source(run, *, check_raw=False):
    run = Path(run).resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if (not manifest.get("episodes") or manifest.get("selection_policy")
            or manifest.get("pipeline_stage") != "evidence"):
        raise ValueError("请选择本流程的完整原始证据检查目录；历史裁剪结果不能用作原件。")
    if digest(manifest["config"]) != manifest["config_sha256"]:
        raise ValueError("Source audit configuration changed")
    fingerprints = {}
    for e in manifest["episodes"]:
        if not e.get("structural_pass") or not e.get("video_verified"):
            raise ValueError(f"Completed structure/video audit required: {e['key']}")
        if sha(inside(run, run / e["index_path"])) != e["index_sha256"]:
            raise ValueError(f"Source evidence index changed: {e['key']}")
        if not e.get("source_ref"):
            raise ValueError(f"Missing immutable raw source reference: {e['key']}")
        fingerprints.update({f["path"]: f for f in e["source_fingerprints"]})
    if not fingerprints:
        raise ValueError("Missing raw source fingerprints")
    if check_raw:
        check_fingerprints(fingerprints.values())
    return manifest


def build(run, output, config=None):
    """New parameters always reconstruct facts; old threshold-derived flags
    are never reused. Video decoding integrity is inherited from hashed inputs.
    """
    run = Path(run).resolve()
    cfg = validate_config(config) if config is not None else load_config()
    parent_sha = sha(run / "manifest.json")
    parent = validate_source(run, check_raw=True)
    output = Path(output).resolve()
    for protected in (run, Path(parent["source_root"]).resolve()):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Output must be separate from raw data and earlier results")
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"run": str(run), "manifest_sha256": parent_sha,
                  "source_fingerprints_sha256": parent["source_fingerprints_sha256"],
                  "raw_integrity_check": "all_original_bytes_rehashed_before_and_after",
                  "video_integrity_check": "inherited_from_unchanged_fully_decoded_raw_video",
                  "manual_reviews": "not_imported_new_result_pending"}
    implementation = implementation_fingerprints()
    write_json(output / "run_started.json", {"configuration": cfg, "parent_audit": provenance})
    write_json(output / "implementation.json", implementation)
    write_json(output / "parameters.json", cfg)
    entries = []
    for original in parent["episodes"]:
        source = load_episode(original["source_ref"])
        if source["load_errors"]:
            raise ValueError(f"Raw source cannot be rebuilt: {original['key']}: {source['load_errors']}")
        selected = derive_episode(source, cfg)
        if len(selected["rows"]) != original["metrics"]["raw_frames"]:
            raise ValueError("Evidence must contain every original frame")
        directory = inside(output, output / "episodes" / original["key"])
        directory.mkdir(parents=True, exist_ok=False)
        write_rows(directory / "frames.jsonl", selected["rows"])
        write_analysis_artifacts(directory, selected)
        entry = copy.deepcopy(original)
        entry["parent_audit_metrics"] = entry.pop("metrics")
        entry["metrics"] = {
            **{k: entry["parent_audit_metrics"][k] for k in (
                "controller_version", "terminal_task_state", "terminal_success",
                "terminal_target_position_error_m", "collision_count", "contact_loss_count")
               if k in entry["parent_audit_metrics"]},
            **selected["metrics"], "human_quality": "pending",
        }
        entry.update(index_path=str((directory / "frames.jsonl").relative_to(output)),
                     index_sha256=sha(directory / "frames.jsonl"),
                     segments=selected["segments"], cuts=selected["cuts"],
                     quality_review="pending", training_approved=False, training_ready=False,
                     parent_index_sha256=original["index_sha256"], source_audit_inherited=True,
                     protected_intervals=selected.get("protected_intervals", []),
                     execution_intervals=selected.get("execution_intervals", []),
                     terminal_release=selected.get("terminal_release"),
                     warnings=sorted(set(original.get("warnings", []) + selected.get("warnings", []))))
        entry["quality_exclusion_reason"] = excluded_episode_reason(cfg, entry["key"])
        entry["training_candidate"] = not bool(entry["quality_exclusion_reason"])
        if selected.get('sequence_contract') is not None:
            entry.update(sequence_path=str((directory / 'sequence.json').relative_to(output)),
                         sequence_sha256=sha(directory / 'sequence.json'),
                         sequence_metrics=selected['sequence_contract']['metrics'])
        shutil.copyfile(inside(run, run / original["io_report_path"]), directory / "io_audit.json")
        entry["io_report_path"] = str((directory / "io_audit.json").relative_to(output))
        entry["analysis_directory"] = str(directory.relative_to(output))
        write_json(directory / "result.json", entry)
        entries.append(entry)
        print(f"[{len(entries)}/{len(parent['episodes'])}] {entry['key']}: "
              f"{len(selected['rows'])} -> {entry['metrics']['kept_frames']}", flush=True)
        del source, selected
    fingerprints = {f["path"]: f for e in entries for f in e["source_fingerprints"]}
    check_fingerprints(fingerprints.values())
    excluded_count = sum(bool(e["quality_exclusion_reason"]) for e in entries)
    summary = {"episodes": len(entries), "structural_pass": len(entries),
               "human_pending": len(entries) - excluded_count,
               "explicitly_excluded_episodes": excluded_count,
               "training_candidates": len(entries) - excluded_count}
    for key in ("raw_frames", "kept_frames", "filtered_frames", "segment_count",
                "task_segment_count", "soft_gap_count", "grasp_protected_frames",
                "release_protected_frames", "height_protected_frames", "execution_grasp_frames",
                "execution_release_frames", "ambiguous_frames", "same_intent_idle_frames",
                "recenter_idle_frames", "intent_transition_idle_frames", "intent_transition_hold_frames",
                "terminal_tail_excluded_frames"):
        summary[key] = sum(e["metrics"].get(key, 0) for e in entries)
    summary["removed_fraction"] = summary["filtered_frames"] / max(1, summary["raw_frames"])
    from dataset.task3.sequence_stitching import VERSION as SEQUENCE_VERSION
    for key in ('safe_stitch_count', 'hard_gap_count', 'boundary_restored_frames',
                'full_valid_action_windows_30', 'sequence_count'):
        summary[key] = sum(e.get('sequence_metrics', {}).get(key, 0) for e in entries)
    from tasks.artifacts import fields
    manifest = {**fields(parent), "version": parent["version"], "pipeline_stage": "curation",
        'sequence_contract': SEQUENCE_VERSION,
        "selection_policy": POLICY, "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(), "source_root": parent["source_root"],
        "config": cfg, "config_sha256": digest(cfg),
        "effective_config_sha256": digest(effective_config(cfg)),
        "source_fingerprints_sha256": parent["source_fingerprints_sha256"],
        "implementation_sha256": implementation, "parent_audit": provenance,
        "episodes": entries, "partial_scope": parent.get("partial_scope", False),
        "training_approved": False, "training_ready": False, "executor_labels_frozen": False,
        "training_compatibility": {"status": "requires_prepared_and_pack",
            "reason": "Export Prepared and verify Training Pack; training.papera reads the certified virtual sequences and causal action masks."},
        "summary": summary}
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        columns = ("key", "raw_frames", "kept_frames", "filtered_frames", "segment_count",
                   "height_protected_frames", "terminal_tail_excluded_frames", "ambiguous_frames",
                   "training_candidate", "quality_exclusion_reason")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for entry in entries:
            writer.writerow({k: (entry | entry["metrics"]).get(k) for k in columns})
    # The manifest is a completion marker; exceptions leave a non-selectable run.
    if sha(run / "manifest.json") != parent_sha:
        raise ValueError("Source manifest changed during curation")
    if implementation_fingerprints() != implementation:
        raise ValueError("Processing source code changed during curation; retry with a fixed implementation")
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False), flush=True)
    return output, manifest
