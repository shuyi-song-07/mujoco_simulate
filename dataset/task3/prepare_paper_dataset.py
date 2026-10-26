"""Raw evidence audit, reversible human review and Prepared export.

Internal engine. Daily entry: dataset/START-DATASET.cmd or manage_dataset.py.
No command changes raw episodes or starts training.
"""
from __future__ import annotations

from dataset.task3.review_sources import select_review_sources

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from dataset.task3.curation_io import discover_episodes, load_episode, audit_episode, fingerprint_file
from dataset.task3.curation_rules import load_config, validate_config, effective_config
from dataset.task3.curation_config import excluded_episode_reason
from dataset.task3.frame_evidence import build_frame_evidence
from dataset.task3.review_server import load_reviews, serve_review, validate_frame_overrides


VERSION = "papera-curation-run-v1"
SIDES = ("left", "right")
CATEGORIES = ("ILoveYou", "Open_Palm", "Closed_Fist")
POLICY = "papera-intent-aware"


def serial(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False, default=serial).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False, default=serial) + "\n", encoding="utf-8")


def write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                                   allow_nan=False, default=serial) + "\n")


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def inside(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Expected file inside {root}: {path}")
    return path


def new_output(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from, and cannot contain, the raw source directory")
    output.mkdir(parents=True, exist_ok=False)
    return output


def check_fingerprints(fingerprints):
    """Re-hash (not only stat) before accepting a human-reviewed derivation."""
    for old in fingerprints:
        current = fingerprint_file(Path(old["path"]), force=True)
        if any(current[k] != old[k] for k in ("sha256", "size", "mtime_ns")):
            raise ValueError(f"Source changed since audit; create a new audit: {old['path']}")


def implementation_fingerprints():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "dataset/task3").glob("*.py"))
    paths.append(root / "dataset/manage_dataset.py")
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def write_analysis_artifacts(directory, derived):
    """Readable facts/decisions with raw indexes; never a compressed time axis."""
    directory = Path(directory)
    timeline = []
    for row in derived["rows"]:
        modes = (row.get("original_control_labels") or {}).get("interaction_modes") or {}
        timeline.append({
            "raw_frame_index": row["raw_frame_index"], "sim_time": row["original_time_s"],
            "source_control_tick": row.get("source_control_tick"),
            "control_epoch": row.get("control_epoch"),
            "left_control_mode": row.get("accepted_left_intent", modes.get("left")),
            "right_control_mode": row.get("accepted_right_intent", modes.get("right")),
            "gripper_command": [(row.get("action_recorded") or [None] * 14)[i] for i in (6, 13)],
            "gripper_actual_position": [(row.get("state_recorded") or [None] * 16)[i] for i in (7, 15)],
            "task_state": row.get("status_summary", {}).get("task_state"),
            "source_refs": row.get("source_refs"),
            "physical_and_intent_evidence": row.get("evidence", {}),
            "frame_role": row.get("frame_role"), "policy_decision": row.get("policy_decision"),
            "policy_reason": row.get("policy_reason"),
        })
    write_json(directory / "raw_event_timeline.json", timeline)
    if timeline:
        with (directory / "raw_event_timeline.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(timeline[0]))
            writer.writeheader()
            for record in timeline:
                writer.writerow({k: json.dumps(v, ensure_ascii=False, default=serial)
                                 if isinstance(v, (dict, list)) else v for k, v in record.items()})
    for name in ("events", "stagnant_runs", "protected_intervals", "execution_intervals"):
        write_json(directory / f"{name}.json", derived.get(name, []))
    if derived.get('sequence_contract') is not None:
        write_json(directory / 'sequence.json', derived['sequence_contract'])
    write_json(directory / "terminal_release.json", derived.get("terminal_release"))
    from collections import Counter
    write_json(directory / "role_summary.json",
               dict(Counter(row.get("frame_role", "EVIDENCE_ONLY") for row in derived["rows"])))


def quality_metrics(episode):
    frames = episode.get("frames", [])
    status = (episode.get("terminal") or {}).get("status") or {}
    out = {"controller_version": episode.get("metadata", {}).get("gripper_trigger_policy"),
           "terminal_task_state": status.get("task_state"),
           "terminal_success": status.get("success"), "human_quality": "pending"}
    for field in ("collision_count", "contact_loss_count"):
        out[field] = status.get(field)
    for field in ("object_position", "target_position", "target_orientation_error_rad"):
        out[field] = status.get(field)
    try:
        out["terminal_target_position_error_m"] = float(np.linalg.norm(
            np.asarray(status["object_position"]) - np.asarray(status["target_position"])))
    except (KeyError, TypeError, ValueError):
        out["terminal_target_position_error_m"] = None
    out["recorded_metrics"] = status.get("metrics", {})
    out["paused_raw_frames"] = sum(bool(f.get("status", {}).get("paused")) for f in frames)
    out["done_raw_frames"] = sum(f.get("status", {}).get("task_state") == "DONE" for f in frames)
    out["review_note"] = "Success/contact/collision counters are evidence, not an automatic quality grade."
    return out


def _audit_one(ref, config, output, check_videos):
    directory = output / "episodes" / ref["key"]
    directory.mkdir(parents=True, exist_ok=False)
    entry = {"key": ref["key"], "session": ref.get("session"),
             "episode_index": ref["episode_index"], "dataset_root": ref["dataset_root"],
             "index_path": str((directory / "frames.jsonl").relative_to(output)),
             "io_report_path": str((directory / "io_audit.json").relative_to(output)),
             "quality_review": "pending", "training_approved": False}
    try:
        episode = load_episode(ref)
        report = audit_episode(episode, check_videos=check_videos)
        entry.update({k: report[k] for k in ("structural_pass", "structural_data_pass", "video_verified",
                     "errors", "warnings", "source_fingerprints")})
        entry["video_sources"] = episode["video_sources"]
        entry['review_video_sources'] = episode.get('review_video_sources',
            select_review_sources(episode['video_sources']))
        entry["controller_version"] = episode["metadata"].get("gripper_trigger_policy")
        entry["source_ref"] = ref
        entry["metrics"] = report["metrics"] | quality_metrics(episode)
        try:
            # An audit contains all facts. The only keep/remove policy runs in
            # rebuild_curation, after source integrity has passed.
            from tasks.registry import hook
            derived = hook(episode['metadata']['config'], 'dataset_adapter')().build_evidence(episode, config)
            write_rows(directory / "frames.jsonl", derived["rows"])
            write_analysis_artifacts(directory, derived)
            entry.update(segments=[], cuts=[], analysis_directory=str(directory.relative_to(output)))
            entry["metrics"].update(raw_frames=len(derived["rows"]),
                                    kept_frames=len(derived["rows"]), filtered_frames=0, segment_count=0)
            entry["warnings"] = sorted(set(entry["warnings"] + derived.get("warnings", [])))
        except Exception as exc:
            entry["structural_pass"] = False
            entry["errors"].append(f"derivation: {type(exc).__name__}: {exc}")
            write_rows(directory / "frames.jsonl", [])
        write_json(directory / "io_audit.json", report)
    except Exception as exc:
        entry.update(structural_pass=False, structural_data_pass=False, video_verified=False,
                     errors=[f"audit: {type(exc).__name__}: {exc}"], warnings=[], video_sources=[],
                     source_fingerprints=[], metrics={"raw_frames": 0}, segments=[], cuts=[])
        write_rows(directory / "frames.jsonl", [])
        write_json(directory / "io_audit.json", {"error": str(exc), "traceback": traceback.format_exc()})
    entry["index_sha256"] = hashlib.sha256((directory / "frames.jsonl").read_bytes()).hexdigest()
    write_json(directory / "result.json", entry)
    return entry


def audit(source, output, config=None, workers=2, check_videos=True, limit=None, episode_key=None, *, engineering=False):
    source = Path(source).resolve()
    refs = discover_episodes(source, engineering=engineering)
    if episode_key:
        refs = [r for r in refs if r["key"] == episode_key]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        refs = refs[:limit]
    if not refs:
        raise ValueError("No saved-success episodes matched; nothing was written")
    if not 1 <= workers <= 4:
        raise ValueError("workers must be 1..4 to bound research-log memory")
    cfg = validate_config(config) if isinstance(config, dict) else load_config(config)
    implementation = implementation_fingerprints()
    output = new_output(source, output)
    write_json(output / "run_started.json", {"version": VERSION, "source_root": source,
        "configuration": cfg, "requested_episodes": [r["key"] for r in refs]})
    entries = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_audit_one, ref, cfg, output, check_videos): ref for ref in refs}
        for done in as_completed(futures):
            entry = done.result(); entries.append(entry)
            print(f"[{len(entries)}/{len(refs)}] {'PASS' if entry['structural_pass'] else 'CHECK'} "
                  f"{entry['key']} raw={entry['metrics'].get('raw_frames')} "
                  f"kept={entry['metrics'].get('kept_frames')} errors={len(entry['errors'])}", flush=True)
    entries.sort(key=lambda e: e["key"])
    source_hashes = {f["path"]: f for ep in entries for f in ep["source_fingerprints"]}
    # Recheck stat after the entire run; per-file hashes were read against the
    # pre-load stat in the IO layer. No episode is silently skipped on failure.
    for ep in entries:
        for f in ep["source_fingerprints"]:
            try:
                st = Path(f["path"]).stat()
                if st.st_size != f["size"] or st.st_mtime_ns != f["mtime_ns"]:
                    raise ValueError("modified during audit")
            except (OSError, ValueError) as exc:
                ep["structural_pass"] = False
                ep["errors"].append(f"source_changed_during_run: {f['path']}: {exc}")
        write_json(output / "episodes" / ep["key"] / "result.json", ep)
    from tasks.artifacts import source_contract
    manifest = {**source_contract(refs), "version": VERSION, "pipeline_stage": "evidence", "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(), "source_root": source,
        "config": cfg, "config_sha256": digest(cfg),
        "effective_config_sha256": digest(effective_config(cfg)),
        "implementation_sha256": implementation,
        "source_fingerprints_sha256": digest(sorted(source_hashes.values(), key=lambda f: f["path"])),
        "episodes": entries, "partial_scope": limit is not None or episode_key is not None,
        "training_approved": False, "training_ready": False, "executor_labels_frozen": False,
        "summary": {"episodes": len(entries), "structural_pass": sum(e["structural_pass"] for e in entries),
            "human_pending": len(entries), "source_files": len(source_hashes),
            **{k: sum(e["metrics"].get(k, 0) for e in entries) for k in
               ("raw_frames", "kept_frames", "filtered_frames", "segment_count")}}}
    write_json(output / "parameters.json", cfg)
    write_json(output / "implementation.json", manifest["implementation_sha256"])
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        columns = ("key", "controller_version", "structural_pass", "raw_frames", "kept_frames",
                   "filtered_frames", "segment_count", "terminal_target_position_error_m", "human_quality")
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for ep in entries:
            values = ep | ep["metrics"]
            writer.writerow({key: values.get(key) for key in columns})
    if implementation_fingerprints() != implementation:
        raise ValueError("Processing source code changed during audit; retry with a fixed implementation")
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest["summary"], ensure_ascii=False), flush=True)
    return manifest


def apply_review(rows, review):
    """Apply reversible manual intervals and re-segment without erasing cuts."""
    validate_frame_overrides(rows, review)
    rows = copy.deepcopy(rows)
    if rows and "policy_decision" in rows[0]:
        from dataset.task3.curation_rules import resegment_rows
        for row in rows:
            raw = row["raw_frame_index"]
            excluded = any(a <= raw < b for a, b in review["exclude_intervals"])
            required = any(a <= raw < b for a, b in review["keep_intervals"])
            row["human_decision"] = "exclude" if excluded else "keep" if required else None
            row["human_reason"] = review.get("notes", "") if excluded or required else None
            row["keep"] = False if excluded else True if required else row["keep"]
            row["effective_keep"] = row["keep"]
            # policy_reason and physical evidence never change on manual edits.
            row["reason"] = list(row["policy_reason"])
            row["review_status"] = review["quality"]
            row["curation_approved"] = review["quality"] in ("A", "B") and bool(review.get("watched_full"))
            row["training_approved"] = False
            row["training_ready"] = False
        resegment_rows(rows)
        return rows
    # Frozen historical training artifacts retain their old manual-interval
    # semantics; this is not an old automatic curation strategy or daily entry.
    segment, previous, old_segment = -1, None, None
    for row in rows:
        raw = row["raw_frame_index"]; old = row.get("segment_id")
        if any(a <= raw < b for a, b in review["exclude_intervals"]):
            row["keep"] = False; row["reason"] = ["human_excluded"]
        elif any(a <= raw < b for a, b in review["keep_intervals"]):
            if "invalid_numeric" in row["reason"]:
                raise ValueError("Human keep cannot override invalid numeric evidence")
            row["keep"] = True; row["reason"].append("human_required_context")
        if not row["keep"]:
            row["segment_id"] = None; previous = None; old_segment = None
            continue
        boundary = (previous is None or raw != previous["raw_frame_index"] + 1 or
            row.get("control_epoch") != previous.get("control_epoch") or
            row.get("session") != previous.get("session") or
            bool(set(row.get("cut_reasons", [])) - {"first_retained_frame", "filtered_gap"}) or
            (old is not None and old_segment is not None and old != old_segment) or
            not 0 < row["original_time_s"] - previous["original_time_s"] <= .05)
        if boundary:
            segment += 1
        row["segment_id"] = segment
        row["review_status"] = review["quality"]
        row["curation_approved"] = True
        row["training_approved"] = False  # Executor/target contract remains a separate gate.
        previous, old_segment = row, old
    return rows


def normalized_geometry(rows, measurements):
    """Keep both raw coordinate systems and a wrist-relative, scale-normalized target.

    These are monocular model estimates, not calibrated robot/world coordinates.
    Invalid/missing targets are zero-filled with explicit masks, never class 0.
    """
    n = len(rows)
    arrays = {name: np.zeros((n, 2, 21, 3), np.float32) for name in
              ("hand_image", "hand_model_world", "hand_normalized")}
    arrays.update({name: np.zeros((n, 2), bool) for name in
                   ("category_mask", "geometry_mask", "task_geometry_mask", "raw_geometry_present")})
    arrays["category_id"] = np.full((n, 2), -1, np.int16)
    for i, row in enumerate(rows):
        for j, side in enumerate(SIDES):
            aux = row["auxiliary"][side]
            for name in ("category_mask", "geometry_mask", "task_geometry_mask"):
                arrays[name][i, j] = aux[name]
            if aux.get("category") in CATEGORIES:
                arrays["category_id"][i, j] = CATEGORIES.index(aux["category"])
            measurement = measurements.get(aux.get("measurement_id"), {})
            hands = [h for h in measurement.get("hands", []) if h.get("bound_side") == side]
            if len(hands) != 1:
                if aux["geometry_mask"]:
                    raise ValueError("Geometry mask references absent/ambiguous hand")
                continue
            hand = hands[0]
            image = np.asarray(hand.get("image_landmarks_21", []), dtype=float)
            world = np.asarray(hand.get("hand_center_landmarks_m_21", []), dtype=float)
            if image.shape != (21, 3) or world.shape != (21, 3) or not np.isfinite(image).all() or not np.isfinite(world).all():
                if aux["geometry_mask"]:
                    raise ValueError("Geometry mask references malformed landmarks")
                continue
            arrays["hand_image"][i, j] = image
            arrays["hand_model_world"][i, j] = world
            arrays["raw_geometry_present"][i, j] = True
            scale = float(np.linalg.norm(world[5] - world[17]))
            if scale > 1e-6:
                arrays["hand_normalized"][i, j] = (world - world[0]) / scale
            elif aux["geometry_mask"]:
                raise ValueError("Invalid palm width in supervised geometry")
    return arrays


def optional_file_sha(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def prepared_rows(rows, review, review_mode):
    """Use the frozen policy once; retain explicit human exclusions in auto mode.

    Pending human keeps cannot restore automatically removed frames. A completed
    A/B review can apply both keeps and exclusions in either export mode.
    """
    rows = copy.deepcopy(rows)
    for row in rows:
        row.pop('human_decision', None)
        row.pop('human_reason', None)
        row['keep'] = row.get('policy_decision') == 'keep'
        row['effective_keep'] = row['keep']
    human_reviewed = review.get('quality') in ('A', 'B') and review.get('watched_full') is True
    effective_review = copy.deepcopy(review)
    if not human_reviewed:
        effective_review['keep_intervals'] = []
    if human_reviewed or effective_review.get('exclude_intervals'):
        rows = apply_review(rows, effective_review)
    else:
        from dataset.task3.curation_rules import resegment_rows
        resegment_rows(rows)
    for row in rows:
        row['review_status'] = review['quality'] if human_reviewed else 'unreviewed'
        row['review_mode'] = review_mode
        row['human_reviewed'] = human_reviewed
        row['curation_approved'] = human_reviewed
        row['training_candidate'] = bool(row['keep'])
        row['training_ready'] = False
    return rows


def prepare(run, output, *, review_mode='reviewed'):
    if review_mode not in ('auto', 'reviewed'):
        raise ValueError('review_mode must be auto or reviewed')
    run = Path(run).resolve()
    manifest_bytes = (run / "manifest.json").read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("version") != VERSION or manifest.get("selection_policy") != POLICY
            or manifest.get("pipeline_stage") != "curation"):
        raise ValueError("历史结果仅只读；请处理原件生成当前裁剪结果后再导出Prepared。")
    validate_config(manifest["config"])
    if (digest(manifest["config"]) != manifest["config_sha256"]
            or digest(effective_config(manifest["config"])) != manifest["effective_config_sha256"]):
        raise ValueError("Curation configuration changed")
    implementation = implementation_fingerprints()
    review_path = run / "reviews.json"
    review_sha = optional_file_sha(review_path)
    reviews = load_reviews(run, manifest)
    selected, selection = [], []
    for ep in manifest['episodes']:
        review = reviews['episodes'].get(ep['key'], {'quality': 'pending', 'watched_full': False,
            'exclude_intervals': [], 'keep_intervals': [], 'notes': ''})
        reason = (excluded_episode_reason(manifest['config'], ep['key'])
                  or ep.get('quality_exclusion_reason'))
        if reason:
            reason = 'explicit_quality_exclusion: ' + reason
        elif review['quality'] in ('C', 'rejected'):
            reason = 'human_excluded_quality'
        elif not ep.get('structural_pass'):
            reason = 'structural_or_alignment_failure'
        elif review_mode == 'reviewed' and not (
                review['quality'] in ('A', 'B') and review.get('watched_full') is True):
            reason = 'not_fully_reviewed_AB'
        selection.append({'key': ep['key'], 'included': reason is None,
                          'reason': reason, 'quality': review['quality']})
        if reason is None:
            selected.append((ep, review))
    if not selected:
        if review_mode == 'reviewed':
            raise ValueError('No A/B episodes approved after full video review; no output written.')
        raise ValueError('No structurally valid, non-excluded automatic candidates; no output written.')
    # Validate all sources and derived indexes before creating the destination.
    unique = {f["path"]: f for e, _ in selected for f in e["source_fingerprints"]}
    check_fingerprints(unique.values())
    if optional_file_sha(review_path) != review_sha:
        raise ValueError("Review changed during preparation preflight; retry with a stable review")
    ready = []
    for ep, review in selected:
        path = inside(run, run / ep["index_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != ep["index_sha256"]:
            raise ValueError(f"Audit frame index changed: {ep['key']}")
        validate_frame_overrides(read_rows(path), review)
        views = [v for v in ep['video_sources'] if v.get('camera') == 'oblique']
        if len(views) != 1:
            raise ValueError(f"Expected one oblique training video: {ep['key']}; convert historical RGB first")
        if not any(r['keep'] for r in prepared_rows(read_rows(path), review, review_mode)):
            decision = next(x for x in selection if x['key'] == ep['key'])
            decision.update(included=False, reason='no_frames_after_manual_exclusion')
            continue
        ready.append((ep,review))
    selected = ready
    if not selected:
        raise ValueError('No retained episodes after manual exclusions; no output written')
    output = Path(output).resolve()
    if output == run or run in output.parents or output in run.parents:
        raise ValueError("Prepared output must be separate from immutable audit run")
    output = new_output(manifest["source_root"], output)
    outputs = []
    for ep, review in selected:
        rows = prepared_rows(read_rows(run / ep["index_path"]), review, review_mode)
        from dataset.task3.sequence_stitching import build_sequence_contract
        sequence = build_sequence_contract(rows, manifest['config'], restore_boundaries=True)
        rows = sequence.pop('rows')
        from dataset.task3.curation_rules import resegment_rows
        resegment_rows(rows)
        kept = [r for r in rows if r["keep"]]
        if not kept:
            raise ValueError(f"No retained frames: {ep['key']}")
        if any(r.get('virtual_clock_valid') is False for r in kept):
            raise ValueError(f"Frozen 30 Hz source clock missing: {ep['key']}; keep in curation for review")
        source = load_episode(ep["source_ref"])
        if source["load_errors"]:
            raise ValueError(f"Source reload failed: {source['load_errors']}")
        arrays = normalized_geometry(kept, source["measurements"])
        arrays.update(state=np.asarray([r["state_recorded"] for r in kept], np.float32),
            action_recorded=np.asarray([r["action_recorded"] for r in kept], np.float32),
            next_control_target=np.asarray([r["next_control_target"]["action"] if r["next_control_target"] else
                                            [0.] * 14 for r in kept], np.float32),
            next_control_target_mask=np.asarray([bool(r["next_control_target"]) for r in kept]),
            raw_frame_index=np.asarray([r["raw_frame_index"] for r in kept], np.int64),
            segment_id=np.asarray([r["segment_id"] for r in kept], np.int64),
            task_segment_id=np.asarray([r["task_segment_id"] for r in kept], np.int64),
            contiguous_fragment_id=np.asarray([r["contiguous_fragment_id"] for r in kept], np.int64),
            source_control_tick=np.asarray([r.get("source_control_tick") if r.get("source_control_tick") is not None else -1 for r in kept], np.int64),
            source_control_tick_mask=np.asarray([r.get("source_control_tick") is not None for r in kept], bool),
            control_epoch=np.asarray([r.get("control_epoch") if r.get("control_epoch") is not None else -1 for r in kept], np.int64),
            control_epoch_mask=np.asarray([r.get("control_epoch") is not None for r in kept], bool),
            original_time_s=np.asarray([r["original_time_s"] for r in kept], np.float64),
            training_frame_index=np.asarray([r['training_frame_index'] for r in kept], np.int64),
            training_time_s=np.asarray([r['training_time_s'] for r in kept], np.float64),
            train_sequence_id=np.asarray([r['train_sequence_id'] for r in kept], np.int64),
            sequence_frame_index=np.asarray([r['sequence_frame_index'] for r in kept], np.int64),
            sequence_time_s=np.asarray([r['sequence_time_s'] for r in kept], np.float64),
            causal_target_valid=np.asarray([r['causal_target_valid'] for r in kept], bool))
        directory = output / "episodes" / ep["key"]; directory.mkdir(parents=True)
        np.savez_compressed(directory / "arrays.npz", **arrays)
        write_rows(directory / "frames.jsonl", rows)
        write_json(directory / 'sequence.json', sequence)
        human_reviewed = review.get('quality') in ('A', 'B') and review.get('watched_full') is True
        training_sources = [v for v in ep['video_sources'] if v.get('camera') == 'oblique']
        review_sources = ep.get('review_video_sources', select_review_sources(ep['video_sources']))
        outputs.append({"key": ep["key"], "session": ep["session"], "review": review,
            'review_mode': review_mode, 'human_reviewed': human_reviewed,
            'review_status': review['quality'] if human_reviewed else 'unreviewed',
            "arrays_path": str((directory / "arrays.npz").relative_to(output)),
            "index_path": str((directory / "frames.jsonl").relative_to(output)),
            'sequence_path': str((directory / 'sequence.json').relative_to(output)),
            'sequence_sha256': hashlib.sha256((directory / 'sequence.json').read_bytes()).hexdigest(),
            'sequence_metrics': sequence['metrics'],
            "video_sources": training_sources, 'training_video_sources': training_sources,
            'review_video_sources': review_sources, "kept_frames": len(kept),
            'raw_frames': len(rows), 'training_candidate': True,
            "source_ref": ep["source_ref"], "source_fingerprints": ep["source_fingerprints"],
            "index_sha256": hashlib.sha256((directory / "frames.jsonl").read_bytes()).hexdigest(),
            "arrays_sha256": hashlib.sha256((directory / "arrays.npz").read_bytes()).hexdigest(),
            "curation_approved": human_reviewed, "training_approved": False, "training_ready": False})
        print(f"Prepared {ep['key']}: {len(kept)} frames", flush=True)
    # Fail if any source changed while normalized targets were being derived.
    check_fingerprints(unique.values())
    if optional_file_sha(run / 'reviews.json') != review_sha:
        raise ValueError("Review changed during preparation; this incomplete output has no completion manifest")
    if hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest() != manifest_sha:
        raise ValueError("Audit manifest changed during preparation")
    for ep, _ in selected:
        if hashlib.sha256((run / ep["index_path"]).read_bytes()).hexdigest() != ep["index_sha256"]:
            raise ValueError(f"Audit index changed during preparation: {ep['key']}")
    human_reviewed = all(e['human_reviewed'] for e in outputs)
    from dataset.task3.sequence_stitching import VERSION as SEQUENCE_VERSION
    from tasks.artifacts import fields
    prepared = {**fields(manifest), "version": "papera-prepared-single-view-v2", 'pipeline_stage': 'prepared',
        'sequence_contract': SEQUENCE_VERSION,
        'complete': True, 'review_mode': review_mode, 'human_reviewed': human_reviewed,
        'selection': selection, "selection_policy": POLICY, "audit_run": str(run),
        "source_manifest_sha256": manifest_sha, "reviews_sha256": review_sha,
        "source_fingerprints_sha256": manifest["source_fingerprints_sha256"],
        "implementation_sha256": implementation,
        "review_binding": reviews["binding"], "config": manifest["config"],
        "config_sha256": manifest["config_sha256"],
        "effective_config_sha256": manifest["effective_config_sha256"],
        "categories": list(CATEGORIES), "hand_order": list(SIDES),
        "geometry_normalization": "(model_world_landmark - wrist_0) / distance(5,17); not robot-world coordinates",
        "action_alignment": {"action_recorded": "issued target before current post-step snapshot",
            "next_control_target": "immediately following legal 60Hz control tick in the original epoch, never a compressed-array tick"},
        'cameras': ['oblique'], 'state_dim': 16, 'action_dim': 14,
        "episodes": outputs, "curation_approved": human_reviewed,
        'training_candidate': not manifest.get('engineering_only', False), "training_approved": False, "training_ready": False,
        "training_compatibility": {"status": "requires_training_pack",
            "reason": "Build and verify Training Pack; training.papera consumes its Safe Stitch virtual sequences and masks."},
        "remaining_gate": "Prepared is not a model input. Export Training Pack and complete data quality review.",
        "split_rule": "Split by original episode/session before generating windows; fit statistics on train only"}
    write_json(output / "parameters.json", manifest["config"])
    if implementation_fingerprints() != implementation:
        raise ValueError("Processing source code changed during export; retry with a fixed implementation")
    write_json(output / "implementation.json", implementation)
    write_json(output / "manifest.json", prepared)
    return prepared
