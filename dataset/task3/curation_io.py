"""Read-only, per-episode inputs and integrity checks for Paper A curation.

Structural verification is distinct from human quality review.  Shared LeRobot
shards are addressed by their episode index and time ranges, never file names
inferred from the episode number.  Caches contain only data tied to file stats;
source files are never opened for writing.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq

from dataset.task3.validate_recordings import audit_hand_timeline, read_snapshots
from dataset.task3.source_paths import dataset_path
from simulation.mujoco.dual_arm.action_contract import ACTION_NAMES_V2, STATE_NAMES, V2

CAMERAS = ("top", "front", "side")  # Legacy three-camera layout.
SINGLE_VIEW_CAMERAS = ("oblique",)
COLLECTION_CAMERAS = (*CAMERAS, *SINGLE_VIEW_CAMERAS)
CURRENT_THREE_CAMERAS = ('top', 'front', 'oblique')
SUPPORTED_CAMERA_LAYOUTS = (CURRENT_THREE_CAMERAS, CAMERAS, SINGLE_VIEW_CAMERAS, COLLECTION_CAMERAS)
SIDES = ("left", "right")
POLICIES = {"palm_fist_edge_v1", "palm_fist_edge_v2", "palm_fist_level_v2"}
AUDIT_VERSION = "papera-curation-io-v2-camera-layout"
_HASH_CACHE: dict[tuple, str] = {}
_VIDEO_CACHE: dict[tuple, dict] = {}
_TABLE_CACHE: dict[tuple, list[dict]] = {}


def camera_layout(info: dict) -> tuple[str, ...]:
    """Resolve the *derived dataset's* RGB layout from its LeRobot features.

    Frozen task3 metadata describes the original collection; it must not be
    rewritten just because the same physical episode was re-rendered. Accept
    the original three, derived oblique, or native four-camera layout; never
    silently ignore an unknown or partially migrated camera configuration.
    """
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("LeRobot info.features must be an object")
    prefix = "observation.images."
    names = {key[len(prefix):] for key in features if key.startswith(prefix)}
    for cameras in SUPPORTED_CAMERA_LAYOUTS:
        if names == set(cameras):
            for camera in cameras:
                feature = features[prefix + camera]
                if (not isinstance(feature, dict) or
                        feature.get("dtype") != "video" or
                        feature.get("shape") != [480, 640, 3]):
                    raise ValueError(f"Invalid camera video feature: {camera}")
            return cameras
    raise ValueError(
        f"Unsupported camera layout {sorted(names)}: expected "
        "['top', 'front', 'oblique'], ['oblique'], or historical three/four camera layouts"
    )


def clear_input_caches():
    """Start an independent batch without reusing previous file contents.

    Windows creation time does not identify an in-place rewrite; size and
    modification time alone cannot prove bytes stayed unchanged between runs.
    Call before discovery, while no episode audit workers are running.
    """
    _HASH_CACHE.clear()
    _VIDEO_CACHE.clear()
    _TABLE_CACHE.clear()


def _signature(path: Path) -> tuple:
    s = path.stat()
    if not path.is_file():
        raise ValueError(f"Expected regular file: {path}")
    return (s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino, s.st_dev)


def _inside(root: Path, path: Path) -> Path:
    result = path.resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Source path escapes dataset root: {path}")
    return result


def _json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"JSON object required: {path}")
    return result


def _table(path: Path) -> list[dict]:
    before = _signature(path)
    key = (str(path.resolve()), before)
    if key not in _TABLE_CACHE:
        rows = pq.read_table(path).to_pylist()
        if _signature(path) != before:
            raise ValueError(f"Source changed while reading: {path}")
        _TABLE_CACHE[key] = rows
    return _TABLE_CACHE[key]


def fingerprint_file(path: Path, *, expected_signature: tuple | None = None,
                     force: bool = False) -> dict:
    """Hash complete bytes; force bypasses the within-batch stat cache."""
    path = Path(path).resolve()
    before = _signature(path)
    if expected_signature is not None and before != expected_signature:
        raise ValueError(f"Source changed since loading: {path}")
    key = (str(path), before)
    digest = None if force else _HASH_CACHE.get(key)
    if digest is None:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        digest = h.hexdigest()
    if _signature(path) != before:
        raise ValueError(f"Source changed while hashing: {path}")
    _HASH_CACHE[key] = digest
    return {"path": str(path), "sha256": digest, "size": before[0], "mtime_ns": before[1]}


def discover_episodes(input_root: Path, *, engineering=False) -> list[dict]:
    """Find saved-success candidates, including both edge versions, never approve them.

    Accept a dataset, a directory containing datasets, or the project root.
    Broken/duplicate indexes raise explicitly rather than silently omitting data.
    Unsupported successful schemas remain candidates and fail the actual audit.
    """
    root = Path(input_root).resolve()
    if engineering:
        from tasks.registry import ROOT
        if not root.is_relative_to(ROOT/'.local/checks'):
            raise ValueError('Engineering input must be isolated under .local/checks')
    if (root / "meta/info.json").is_file():
        roots = [root]
    else:
        if (root / "datasets").is_dir():
            root = root / "datasets"
        roots = sorted(p.parent.parent for p in root.glob("*/meta/info.json"))
    result = []
    for dataset in roots:
        dataset = dataset.resolve()
        manifest_path = _inside(dataset, dataset / "research/episodes.jsonl")
        manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = [entry["episode_index"] for entry in manifest]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate research episode index: {dataset}")
        indexes = sorted((dataset / "meta/episodes").rglob("*.parquet"))
        indexed = [row for path in indexes for row in _table(_inside(dataset, path))]
        indexed_ids = [row["episode_index"] for row in indexed]
        if len(indexed_ids) != len(set(indexed_ids)) or set(ids) != set(indexed_ids):
            raise ValueError(f"LeRobot/research episode indexes disagree: {dataset}")
        for entry in manifest:
            if entry.get("outcome") != "success" and not engineering:
                continue
            i = int(entry["episode_index"])
            metadata_path = _inside(dataset, dataset / f"research/episode_{i:06d}_snapshots/episode_metadata.json")
            frozen = _json(metadata_path)
            result.append({"engineering": engineering, "key": f"{dataset.name}/episode_{i:06d}", "dataset_name": dataset.name,
                "dataset_root": dataset, "episode_index": i, "session": frozen.get("session"),
                "metadata_path": metadata_path,
                "research_path": _inside(dataset, dataset / f"research/episode_{i:06d}.jsonl"),
                "outcome": entry.get("outcome"), "trigger_policy": frozen.get("gripper_trigger_policy")})
    return sorted(result, key=lambda r: r["key"])


def _nonfinite(value: Any, prefix: str = "") -> str | None:
    if isinstance(value, float) and not math.isfinite(value):
        return prefix
    if isinstance(value, dict):
        for key, item in value.items():
            found = _nonfinite(item, f"{prefix}.{key}")
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            found = _nonfinite(item, f"{prefix}[{i}]")
            if found is not None:
                return found
    return None


def load_episode(ref: dict) -> dict:
    """Load one episode; stream its research file once through the provenance audit.

    `frames` and `controls` retain original control-frame records.  `measurements`
    maps IDs to raw measurements, independent of masks for any auxiliary task.
    Load failures are kept in `load_errors`, so callers can produce an explicit
    failed report instead of losing a candidate from the review inventory.
    """
    root = Path(ref["dataset_root"]).resolve()
    i = int(ref["episode_index"])
    result = {"ref": ref, "metadata": {}, "snapshots": {}, "frames": [], "controls": [],
        "measurements": {}, "measurement_sources": {}, "accepted_events": {}, "executed_events": [],
        "timeline_events": [], "source_files": [],
        "video_sources": [], "review_video_sources": [], "cameras": (), "parquet_rows": [], "load_errors": [], "timeline_errors": [],
        "timeline_coverage": {}, "_source_stats": {}}
    errors = result["load_errors"]

    def source(path):
        p = _inside(root, Path(path))
        if p not in result["source_files"]:
            result["source_files"].append(p)
            result["_source_stats"][str(p)] = _signature(p)
        return p

    try:
        info = result["info"] = _json(source(root / "meta/info.json"))
        result["cameras"] = camera_layout(info)
        result["dataset_metadata"] = _json(source(root / "meta/task3.json"))
        frozen = result["metadata"] = _json(source(ref["metadata_path"]))
        archive = Path(ref["metadata_path"]).parent
        camera_manifest, camera_index = archive / 'camera_manifest.json', archive / 'camera_frame_index.jsonl'
        if (camera_manifest.exists() or camera_index.exists()
                or 'oblique' in frozen.get('recording_cameras', [])):
            result['camera_sync_manifest'] = _json(source(camera_manifest))
            result['camera_sync_index'] = [json.loads(line) for line in
                source(camera_index).read_text(encoding='utf-8').splitlines() if line.strip()]
        for p in sorted(archive.glob("snapshots_*.npz")):
            source(p)
        result["snapshots"] = read_snapshots(archive)
        result["terminal"] = _json(source(archive / "terminal.json"))
        with np.load(source(archive / "terminal_snapshot.npz"), allow_pickle=False) as z:
            result["terminal_snapshot"] = {key: z[key] for key in z.files}
        entries = [json.loads(line) for line in source(root / "research/episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        matched = [entry for entry in entries if entry.get("episode_index") == i]
        if len(matched) != 1:
            raise ValueError("Research episode index missing or duplicated")
        result["manifest_entry"] = matched[0]
        indexed = [entry for p in sorted((root / "meta/episodes").rglob("*.parquet")) for entry in _table(source(p))]
        matched = [entry for entry in indexed if entry.get("episode_index") == i]
        if len(matched) != 1:
            raise ValueError("LeRobot episode index missing or duplicated")
        ep = result["episode_index_metadata"] = matched[0]
        result["dataset_index_counts"] = {"episodes": len(indexed), "research_episodes": len(entries)}
        # An episode can cross data shards.  Filter by explicit episode_index,
        # then verify its global range; its starting shard alone is insufficient.
        for p in sorted((root / "data").rglob("*.parquet")):
            p = source(p)
            selected = [row for row in _table(p) if row.get("episode_index") == i]
            if selected:
                result["parquet_rows"].extend(copy.deepcopy(selected))
        result["parquet_rows"].sort(key=lambda row: row["index"])
        for camera in result["cameras"]:
            key = f"observation.images.{camera}"
            path = dataset_path(root, info["video_path"], video_key=key,
                chunk_index=ep[f"videos/{key}/chunk_index"], file_index=ep[f"videos/{key}/file_index"])
            result["video_sources"].append({"camera": camera, "video_key": key, "path": source(path),
                "start_s": float(ep[f"videos/{key}/from_timestamp"]),
                "end_s": float(ep[f"videos/{key}/to_timestamp"]),
                "fps": float(info["fps"]), "frame_count": int(ep["length"])})
        from dataset.task3.review_sources import provenance_path, review_sources
        provenance = provenance_path(root)
        if provenance is not None:
            source(provenance)
        naming_proof = root / 'meta/naming_migration.json'
        if naming_proof.is_file():
            source(naming_proof)
        result["review_video_sources"] = review_sources(root, i, expected_frames=int(ep["length"]))
    except Exception as exc:
        errors.append(f"episode_load: {type(exc).__name__}: {exc}")

    try:
        path = source(ref["research_path"])
        before = _signature(path)
        digest = hashlib.sha256()
        last_clock = {}

        def research():
            with path.open("rb") as stream:
                for line_number, line in enumerate(stream, 1):
                    digest.update(line)
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"Research line {line_number}: expected object")
                    # In-memory provenance only: the original bytes remain untouched.
                    # Equal tick/time values do not imply events precede an image.
                    row["source_line"] = line_number
                    bad = _nonfinite(row)
                    if bad is not None:
                        message = f"research_nonfinite:{row.get('kind')}:{bad}"
                        if message not in errors:
                            errors.append(message)
                    kind = row.get("kind")
                    if kind == "hand_measurement":
                        m = row.get("measurement")
                        if isinstance(m, dict):
                            mid = m.get("measurement_id")
                            if mid in result["measurements"] and result["measurements"][mid] != m:
                                errors.append(f"measurement_id_reused:{mid}")
                            if mid:
                                # Conflicting reuse is an audit failure. Keep the
                                # first value so late corrupt input cannot rewrite
                                # the facts attached to an earlier observation.
                                result["measurements"].setdefault(mid, m)
                                result["measurement_sources"].setdefault(mid, {
                                    "source_line": line_number, "session": row.get("session"),
                                    "control_epoch": row.get("control_epoch"),
                                    "wall_time_ms": row.get("wall_time_ms"),
                                    "server_monotonic_ns": row.get("server_monotonic_ns")})
                        elif row.get("schema_valid"):
                            errors.append("trusted_measurement_not_object")
                    elif kind == "request_executed":
                        result["executed_events"].append(row)
                        if (row.get("result") or {}).get("accepted") and row.get("request", {}).get("eventId"):
                            eid = row["request"]["eventId"]
                            if eid in result["accepted_events"]:
                                errors.append(f"accepted_event_id_reused:{eid}")
                            else:
                                result["accepted_events"][eid] = row
                    elif kind == "control_frame":
                        result["controls"].append(row)
                        if row.get("frame_index") is not None:
                            result["frames"].append(row)
                        for name in ("sim_time", "wall_time_ms", "server_monotonic_ns"):
                            value = row.get(name)
                            if value is None:
                                continue
                            if name in last_clock and value < last_clock[name]:
                                message = f"control_clock_regressed:{name}"
                                if message not in errors:
                                    errors.append(message)
                            last_clock[name] = value
                    else:
                        # These event types can explain source changes and rejected
                        # inputs. Large raw hand arrays already live in measurements.
                        compact = {k: row.get(k) for k in
                            ("kind", "source_line", "session", "control_epoch", "tick", "sim_time",
                             "wall_time_ms", "server_monotonic_ns", "event_id", "measurement_id",
                             "reason", "drop_reason", "accepted", "clock_pair", "side", "stage")
                            if k in row}
                        request = row.get("request")
                        if isinstance(request, dict):
                            compact["request"] = {k: request.get(k) for k in
                                ("command", "eventId", "measurement_id", "session", "control_epoch")}
                        result["timeline_events"].append(compact)
                    yield row

        result["timeline_errors"], result["timeline_coverage"] = audit_hand_timeline(research(), result["metadata"])
        if _signature(path) != before:
            raise ValueError("Research log changed while loading")
        _HASH_CACHE[(str(path), before)] = digest.hexdigest()
    except Exception as exc:
        errors.append(f"research_load: {type(exc).__name__}: {exc}")
    return result


def _decode_video(path: Path) -> dict:
    before = _signature(path)
    key = (str(path.resolve()), before)
    if key in _VIDEO_CACHE:
        return _VIDEO_CACHE[key]
    times, sizes = [], set()
    # File handles also make OS access errors distinguishable from decoder errors
    # on Windows/non-ASCII paths.  Every encoded frame is decoded to EOF.
    with path.open("rb") as stream, av.open(stream) as container:
        video = container.streams.video[0]
        rate = float(video.average_rate or 0)
        header_frames = int(video.frames)
        for frame in container.decode(video):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Decoded video frame lacks PTS/time_base")
            times.append(float(frame.pts * frame.time_base))
            sizes.add((frame.width, frame.height))
    if _signature(path) != before:
        raise ValueError("Video changed during full decoding")
    values = np.asarray(times, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError("Empty, nonfinite or non-increasing decoded video PTS")
    if header_frames and header_frames != len(values):
        raise ValueError(f"Header/decode frame count mismatch: {header_frames}/{len(values)}")
    result = {"times": values, "sizes": sizes, "rate": rate, "decoded_frames": len(values)}
    _VIDEO_CACHE[key] = result
    return result


def check_video_source(source: dict) -> dict:
    """Decode a complete shared file, then verify only this indexed episode span."""
    result = {**source, "path": str(source["path"]), "verified": False, "errors": []}
    try:
        v = _decode_video(Path(source["path"]))
        start, end, fps = (float(source[k]) for k in ("start_s", "end_s", "fps"))
        count = int(source["frame_count"])
        if not all(math.isfinite(x) for x in (start, end, fps)) or not 0 <= start < end or fps <= 0:
            raise ValueError("Invalid indexed episode video time range")
        selected = v["times"][(v["times"] >= start - 1e-4) & (v["times"] < end - 1e-4)]
        result.update(decoded_frames=v["decoded_frames"], selected_frames=len(selected),
            decoded_sizes=sorted(map(list, v["sizes"])), decoded_fps=v["rate"],
            first_pts_s=float(selected[0]) if len(selected) else None,
            last_pts_s=float(selected[-1]) if len(selected) else None)
        if v["sizes"] != {(640, 480)} or abs(v["rate"] - fps) > .01:
            raise ValueError("Decoded image dimensions or FPS mismatch")
        if len(selected) != count or abs(end - start - count / fps) > .001:
            raise ValueError(f"Indexed video duration/frame count mismatch: {len(selected)} expected {count}")
        if not np.allclose(selected - start, np.arange(count) / fps, atol=.001, rtol=0):
            raise ValueError("Indexed video PTS do not match episode frame clock")
        result["verified"] = True
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def _historical_advisory(message: str, metadata: dict) -> bool:
    # Only reproducible labels derived from preserved raw events may be rebuilt.
    # Invalid source/authorization/freshness checks are never downgraded here.
    return (metadata.get("gripper_trigger_policy") in {"palm_fist_edge_v1", "palm_fist_edge_v2"} and message in {
        "control label mismatch: intervention_reasons",
        "control label mismatch: intervention_mask",
    })


def audit_episode(episode: dict, check_videos: bool = True) -> dict:
    """Verify structure/provenance; never mark human quality review as passed."""
    ref, m = episode["ref"], episode["metadata"]
    errors = list(episode.get("load_errors", []))
    warnings, advisories = [], []
    video_checks, fingerprints = [], []

    def check(ok, message):
        if not ok and message not in errors:
            errors.append(message)

    for message in episode.get("timeline_errors", []):
        if _historical_advisory(message, m):
            advisories.append({"original_error": message, "disposition": "rebuild_from_raw_required",
                "reason": "Historical derived intervention labels; raw measurements, accepted sources and physical snapshots remain independently audited."})
            warnings.append("historical_derived_label: " + message)
        else:
            check(False, "hand_timeline: " + message)
    metrics = {"raw_frames": len(episode["frames"]), "control_rows": len(episode["controls"]),
        "measurement_rows": len(episode["measurements"]),
        "quality_review": m.get("quality_review", "pending"), "outcome": m.get("outcome"),
        "trigger_policy": m.get("gripper_trigger_policy"), **episode.get("timeline_coverage", {})}
    try:
        info, shared = episode["info"], episode["dataset_metadata"]
        entry, index = episode["manifest_entry"], episode["episode_index_metadata"]
        snap, rows, frames, controls = (episode[k] for k in ("snapshots", "parquet_rows", "frames", "controls"))
        n = len(rows)
        check(n > 0, "empty_episode")
        check(m.get("schema_version") == shared.get("schema_version") == V2, "schema_mismatch")
        check(m.get("gripper_trigger_policy") in POLICIES, "unsupported_gripper_trigger_policy")
        check(m.get("control_interface_version") == "gesture-rate-gripper-v3", "unsupported_control_interface")
        check(info.get("fps") == 30, "requires_30hz")
        check(m.get("sampling_policy", {}).get("version") == "raw_fixed_phase_v2", "raw_sampling_policy_mismatch")
        for key, names in (("action", ACTION_NAMES_V2), ("observation.state", STATE_NAMES)):
            check(info.get("features", {}).get(key, {}).get("names") == names, f"feature_names_mismatch:{key}")
        check(m.get("action_names") == ACTION_NAMES_V2 and m.get("state_names") == STATE_NAMES, "frozen_field_names_mismatch")
        check(m.get("action_units") == ["m", "m", "m", "rad", "rad", "rad", "m"] * 2, "action_units_mismatch")
        check(m.get("training_views") == "opaque", "training_view_style_not_frozen")
        digest = hashlib.sha256(json.dumps(m["config"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        check(digest == m.get("config_hash") == shared.get("config_hash") == entry.get("config_hash"), "frozen_config_hash_mismatch")
        check(m.get("session") == ref.get("session") == entry.get("session"), "session_mismatch")
        engineering = ref.get('engineering', False)
        expected_outcome = entry.get('outcome') if engineering else 'success'
        expected_kind = 'scripted_test' if engineering else 'human_teleop'
        check(m.get("outcome") == entry.get("outcome") == expected_outcome, "outcome_mismatch")
        check(m.get("input_kind") == entry.get("input_kind") == shared.get("input_kind") == expected_kind, "input_provenance_mismatch")
        for name in ("control_interface_version", "hand_supervision_version", "policy_contract_version",
                     "gripper_trigger_policy", "gripper_trigger_semantics", "release_intent_contract",
                     "confirmation_gate", "intent_training_timeline", "interaction_mode_contract"):
            check(m.get(name) == shared.get(name), f"frozen_contract_mismatch:{name}")
        check(episode["dataset_index_counts"]["episodes"] == episode["dataset_index_counts"]["research_episodes"] == info["total_episodes"], "dataset_episode_count_mismatch")
        check(n == index["length"] == entry["frames"] == len(frames) == len(snap["time"]), "episode_frame_count_mismatch")
        if 'camera_sync_manifest' in episode:
            sync = episode['camera_sync_manifest']
            names = m.get('recording_cameras') or m.get('config', {}).get('recording', {}).get('cameras', CAMERAS)
            check(sync.get('version') == 'task3-shared-snapshot-rgb-v1'
                  and sync.get('cameras') == list(names) and sync.get('frames') == n
                  and sync.get('fps') == 30 and sync.get('width') == 640 and sync.get('height') == 480,
                  'camera_snapshot_manifest_mismatch')
            mapping = episode['camera_sync_index']
            check(len(mapping) == n and all(
                row.get('raw_frame_index') == j and row.get('snapshot_index') == j
                and row.get('cameras') == list(names)
                and abs(float(row.get('snapshot_time_s', -1)) - float(snap['time'][j])) <= 1e-9
                and abs(float(row.get('video_time_s', -1)) - j / 30) <= 1e-9
                for j, row in enumerate(mapping)), 'camera_snapshot_clock_mismatch')
            if 'oblique' in names:
                check(isinstance(m.get('render_identity'), dict)
                      and sync.get('render_identity') == m.get('render_identity'),
                      'camera_render_identity_mismatch')
        check([r["frame_index"] for r in rows] == list(range(n)), "parquet_frame_indices_mismatch")
        check([r["frame_index"] for r in frames] == list(range(n)), "research_frame_indices_mismatch")
        check(all(r["episode_index"] == ref["episode_index"] for r in rows), "parquet_episode_mismatch")
        check([r["index"] for r in rows] == list(range(index["dataset_from_index"], index["dataset_to_index"])), "parquet_global_range_mismatch")
        check(np.allclose([r["timestamp"] for r in rows], np.arange(n)/30, atol=1e-4, rtol=0), "parquet_training_clock_mismatch")
        dims = m["model_dimensions"]
        check(all(type(dims[k]) is int and dims[k] > 0 for k in ("nq", "nv", "nu", "nbody")), "invalid_model_dimensions")
        shapes = {"time": (), "qpos": (dims["nq"],), "qvel": (dims["nv"],), "ctrl": (dims["nu"],),
            "body_pos": (dims["nbody"], 3), "body_quat": (dims["nbody"], 4), "state": (16,), "action": (14,)}
        for name, shape in shapes.items():
            check(snap[name].shape == (n, *shape) and np.isfinite(snap[name]).all(), f"snapshot_invalid:{name}")
        for name, column in (("state", "observation.state"), ("action", "action")):
            values = np.asarray([r[column] for r in rows], dtype=float)
            check(values.shape == snap[name].shape and np.isfinite(values).all(), f"parquet_invalid:{column}")
            check(np.allclose(values, snap[name], atol=1e-6, rtol=0), f"parquet_snapshot_mismatch:{name}")
            check(np.allclose([r[name] for r in frames], snap[name], atol=1e-6, rtol=0), f"research_snapshot_mismatch:{name}")
        check(np.all((snap["action"][:, [6, 13]] >= 0) & (snap["action"][:, [6, 13]] <= .080001)), "action_claw_width_out_of_bounds")
        recovered, joint_ids = [], []
        for side in SIDES:
            mapping = m["joint_mapping"][side]
            js, fingers = mapping["qpos_indices"], mapping["finger_qpos_indices"]
            check(len(js) == 7 and len(fingers) == 2 and len(mapping["joint_names"]) == 7, f"joint_mapping_size:{side}")
            check(all(type(j) is int and 0 <= j < dims["nq"] for j in js + fingers), f"joint_mapping_bounds:{side}")
            joint_ids += js + fingers
            recovered.extend(snap["qpos"][:, j] for j in js)
            recovered.append(snap["qpos"][:, fingers].sum(axis=1))
        check(len(set(joint_ids)) == 18, "joint_mapping_overlap")
        adr, dof = m["object_qpos_adr"], m["object_dof_adr"]
        check(type(adr) is int and 0 <= adr <= dims["nq"]-7 and
              not (set(range(adr, adr+7)) & set(joint_ids)), "invalid_object_qpos_mapping")
        check(type(dof) is int and 0 <= dof <= dims["nv"]-6, "invalid_object_dof_mapping")
        check(np.allclose(np.asarray(recovered).T, snap["state"], atol=1e-6, rtol=0), "named_joint_state_mismatch")
        check(np.allclose(snap["time"], [r["sim_time"] for r in frames], atol=1e-9, rtol=0), "research_sim_time_mismatch")
        delta = np.diff(snap["time"])
        check(np.all(delta > 0) and np.max(np.abs(snap["time"] - snap["time"][0] - np.arange(n)/30)) <= .0101, "raw_sim_clock_invalid")
        check(not any(r.get("log_dropped", 0) for r in controls), "research_events_dropped")
        for row in controls:
            aid, mid = row.get("action_event_id"), row.get("action_measurement_id")
            source = episode["accepted_events"].get(aid)
            check(bool(aid) == bool(mid), "incomplete_action_source_ids")
            if aid:
                check(source is not None and source["request"].get("measurement_id") == mid, "action_source_reference_mismatch")
            oid = row.get("observation_measurement_id")
            check(not oid or oid in episode["measurements"], "observation_reference_missing")
            check(np.isfinite(row["state"]).all() and len(row["state"]) == 16, "control_state_invalid")
            check(np.isfinite(row["action"]).all() and len(row["action"]) == 14, "control_action_invalid")
        terminal, ts = episode["terminal"], episode["terminal_snapshot"]
        if not engineering:
            check(terminal["status"]["task_state"] == "DONE" and terminal["status"].get("success") is True, "successful_terminal_state_missing")
        check(terminal.get("outcome") == expected_outcome and terminal.get("last_raw_frame") == n-1, "terminal_episode_identity_mismatch")
        check(not terminal.get("log_dropped"), "terminal_events_dropped")
        for name, shape in shapes.items():
            if name in ("state", "action") and name not in ts:
                continue  # terminal archive holds physical state, not action labels
            check(name in ts and ts[name].shape == shape and np.isfinite(ts[name]).all(), f"terminal_snapshot_invalid:{name}")
        lag = float(ts["time"]) - snap["time"][-1]
        qdelta = float(np.max(np.abs(ts["qpos"] - snap["qpos"][-1])))
        check(abs(lag - terminal["sample_to_terminal_s"]) < 1e-8 and 0 <= lag < .044, "terminal_time_mismatch_or_stale")
        check(abs(qdelta - terminal["max_qpos_delta"]) < 1e-8, "terminal_position_delta_mismatch")
        metrics.update(raw_frames=n, nominal_duration_s=n/30, sim_span_s=float(snap["time"][-1]-snap["time"][0]),
            wall_duration_s=entry.get("wall_duration_s"), terminal_lag_s=lag, terminal_qpos_delta=qdelta,
            dataset_metrics=entry.get("metrics", {}))
    except Exception as exc:
        check(False, f"structural_exception: {type(exc).__name__}: {exc}")
    uncertain = metrics.get("hand_freshness_audit", {}).get("uncertain_checks", 0)
    if uncertain:
        warnings.append(f"historical_freshness_uncertain:{uncertain}; review exact recorded bounds")
    if m.get("quality_review") != "passed":
        warnings.append("human_quality_review_pending; structural verification does not approve training")
    if check_videos:
        video_checks = [check_video_source(source) for source in episode["video_sources"]]
        for video in video_checks:
            for message in video["errors"]:
                check(False, f"video:{video['camera']}:{message}")
        check(tuple(v["camera"] for v in video_checks) == episode.get("cameras") and
              episode.get("cameras") in SUPPORTED_CAMERA_LAYOUTS,
              "video:camera_layout_mismatch")
    else:
        warnings.append("videos_not_fully_decoded; diagnostic_only_not_training_eligible")
    # Hash every file used, including shared metadata/data/video shards. A stat
    # change between loading and hashing invalidates this attempt, even on cache.
    for path in episode["source_files"]:
        try:
            fingerprints.append(fingerprint_file(path, expected_signature=episode["_source_stats"].get(str(path))))
        except Exception as exc:
            check(False, f"source_fingerprint:{type(exc).__name__}:{exc}")
    for path in episode["source_files"]:
        try:
            check(_signature(path) == episode["_source_stats"].get(str(path)), f"source_changed_during_audit:{path}")
        except OSError as exc:
            check(False, f"source_missing_during_audit:{path}:{exc}")
    data_errors = [e for e in errors if not e.startswith("video:")]
    expected_cameras = episode.get("cameras", ())
    video_verified = bool(check_videos and expected_cameras in SUPPORTED_CAMERA_LAYOUTS and
        tuple(v["camera"] for v in video_checks) == expected_cameras and
        all(v["verified"] for v in video_checks))
    return {"audit_version": AUDIT_VERSION, "key": ref["key"], "structural_data_pass": not data_errors,
        "video_verified": video_verified, "complete": bool(check_videos),
        "structural_pass": not errors and video_verified, "errors": errors, "warnings": warnings,
        "historical_label_advisories": advisories, "timeline_errors_original": episode.get("timeline_errors", []),
        "quality_review": m.get("quality_review", "pending"), "quality_approved": False,
        "video_checks": video_checks, "source_fingerprints": fingerprints, "metrics": metrics}
