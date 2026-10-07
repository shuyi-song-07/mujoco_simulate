"""Task 3 audits deliberately exercise v3 shared shards and video segments."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dataset.task3.validate_task3_dataset import ACTION_NAMES, STATE_NAMES, validate_dataset
from evaluation.task3.evaluate_act_rollout import safe_policy_action
from evaluation.task3.evaluate_teleop import evaluate_roots
from training.ACT.train_act_dual_arm import install_training_hooks, numeric_training_stats

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_fixture(root):
    state = [[0.] * 7 + [0.08] + [0.] * 7 + [0.08] for _ in range(6)]
    action = [[0.1, 0., 0.2, 0., 0., 0., 255., 0.1, 0.3, 0.2, 0., 0., 0., 255.] for _ in range(6)]
    for i in range(3, 6):
        state[i][0] = 1.
        action[i][0] = 0.15
    features = {"observation.state": {"dtype": "float32", "shape": [16], "names": STATE_NAMES},
                "action": {"dtype": "float32", "shape": [14], "names": ACTION_NAMES}}
    for camera in ("top", "front", "side"):
        features[f"observation.images.{camera}"] = {"dtype": "video", "shape": [480, 640, 3]}
    info = {"codebase_version": "v3.0", "features": features, "fps": 30, "total_frames": 6, "total_episodes": 2,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    write_json(root / "meta/info.json", info)
    write_json(root / "meta/task3.json", {"schema_version": "task3-v1", "field_order": "left_then_right", "coordinate_frame": "task_frame",
                                          "transforms": {key: np.eye(4).tolist() for key in ("T_WT", "T_WL", "T_WR")},
                                          "cameras": {key: {} for key in ("top", "front", "side")},
                                          "config_hash": "fixture-config", "config": {"cooperation": {"relative_pose_hard_limit_m": .04}}})
    shard = root / "data/chunk-000/file-000.parquet"
    shard.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pydict({"index": list(range(6)), "episode_index": [0] * 3 + [1] * 3,
                                      "frame_index": list(range(3)) * 2, "timestamp": [0., 1 / 30, 2 / 30] * 2,
                                      "observation.state": state, "action": action}), shard)
    metadata, manifests = [], []
    for episode in range(2):
        meta = {"episode_index": episode, "length": 3, "dataset_from_index": episode * 3,
                "dataset_to_index": (episode + 1) * 3, "data/chunk_index": 0, "data/file_index": 0}
        for camera in ("top", "front", "side"):
            prefix = f"videos/observation.images.{camera}"
            meta.update({prefix + "/chunk_index": 0, prefix + "/file_index": 0,
                         prefix + "/from_timestamp": episode * .1, prefix + "/to_timestamp": (episode + 1) * .1})
        metadata.append(meta)
        manifests.append({"episode_index": episode, "outcome": "success", "frames": 3, "seed": episode,
                          "wall_duration_s": .5, "training_duration_s": .1, "config_hash": "fixture-config",
                          "metrics": {"collision_count": 0, "contact_loss_count": 0}, "failure_reason": None})
        logs = [{"wall_time_ms": 1000. + i * 30, "frame_index": i, "training_recordable": True,
                 "control_mode": "XY", "task_state": "DONE" if i == 2 else "PREGRASP", "state": state[episode * 3 + i],
                 "action": action[episode * 3 + i], "events": [], "metrics": {"relative_pose_error_m": 0., "object_position": [i * .001, 0., .1]}}
                for i in range(3)]
        write_jsonl(root / f"research/episode_{episode:06d}.jsonl", logs)
    write_jsonl(root / "research/episodes.jsonl", manifests)
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(metadata), path)
    for camera in ("top", "front", "side"):
        path = root / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
        path.parent.mkdir(parents=True)
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
            for i in range(6):
                frame = av.VideoFrame.from_ndarray(np.full((480, 640, 3), i * 30, dtype=np.uint8), format="rgb24")
                frame.pts, frame.time_base = i, Fraction(1, 30)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)


def fixture_snapshots(root, episode=0):
    """Independent archived physics evidence for one existing fixture episode."""
    path = root / f"research/episode_{episode:06d}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for i, row in enumerate(rows):
        row["sim_time"] = i / 30
    write_jsonl(path, rows)
    state = np.asarray([row["state"] for row in rows])
    qpos = np.zeros((len(rows), 25))
    qpos[:, :7], qpos[:, 9:16] = state[:, :7], state[:, 8:15]
    qpos[:, 7:9] = state[:, [7]] / 2
    qpos[:, 16:18] = state[:, [15]] / 2
    qpos[:, 20], qpos[:, 21] = .371, 1
    snapshot = {"state": state, "action": np.asarray([row["action"] for row in rows]),
                "qpos": qpos, "qvel": np.zeros((len(rows), 24)),
                "time": np.asarray([row["sim_time"] for row in rows])}
    directory = root / f"research/episode_{episode:06d}_snapshots"
    write_json(directory / "task3.json", {"config_hash": "fixture-config"})
    archive = directory / "snapshots_000000.npz"
    np.savez_compressed(archive, **snapshot)
    return path, rows, archive, snapshot


class TestTask3LearningPipeline(unittest.TestCase):
    def setUp(self):
        base = ROOT / ".local/task3-check"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="learning-test-", dir=base)
        self.root = Path(self.temp.name)
        self.assertTrue(self.root.resolve().is_relative_to(base.resolve()))
        make_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_shared_v3_shards_and_segmented_videos(self):
        report = validate_dataset(self.root)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["episode_count"], 2)
        self.assertEqual(report["frame_count"], 6)

    def test_failure_is_rejected_by_default(self):
        path = self.root / "research/episodes.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["outcome"], rows[1]["failure_reason"] = "failure", "operator_abort"
        write_jsonl(path, rows)
        self.assertFalse(validate_dataset(self.root)["valid"])
        self.assertTrue(validate_dataset(self.root, allow_failures=True)["valid"])

    def test_wrong_left_right_field_order_rejected(self):
        path = self.root / "meta/info.json"
        value = json.loads(path.read_text())
        value["features"]["action"]["names"] = ACTION_NAMES[7:] + ACTION_NAMES[:7]
        write_json(path, value)
        self.assertFalse(validate_dataset(self.root, check_videos=False)["valid"])

    def test_confirmation_idle_does_not_enter_training(self):
        path = self.root / "research/episode_000000.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["control_mode"] = "GRASP_CONFIRM"
        write_jsonl(path, rows)
        self.assertFalse(validate_dataset(self.root, check_videos=False)["valid"])

    def test_verified_object_drift_and_velocity_during_confirmation_are_allowed(self):
        path, rows, archive, snapshot = fixture_snapshots(self.root)
        rows[1]["control_mode"] = "GRASP_CONFIRM"
        rows[1]["capture_reason"] = "physical_change"
        write_jsonl(path, rows)
        for evidence in ("slow_object_drift", "velocity"):
            with self.subTest(evidence=evidence):
                snapshot["qpos"][:, 18] = 0
                snapshot["qvel"][:] = 0
                if evidence == "slow_object_drift":
                    snapshot["qpos"][1:, 18] = .0002
                else:
                    snapshot["qvel"][1, 18] = .04
                np.savez_compressed(archive, **snapshot)
                report = validate_dataset(self.root, check_videos=False)
                self.assertTrue(report["valid"], report["errors"])
                self.assertEqual(report["episodes"][0]["physical_confirmation_frames"], [1])
                self.assertTrue(report["episodes"][0]["snapshots_verified"])

    def test_stationary_confirmation_with_forged_reason_is_still_rejected(self):
        path, rows, archive, snapshot = fixture_snapshots(self.root)
        rows[1]["control_mode"] = "RELEASE_CONFIRM"
        for reason in ("stationary_pause", "physical_change"):
            with self.subTest(reason=reason):
                rows[1]["capture_reason"] = reason
                write_jsonl(path, rows)
                report = validate_dataset(self.root, check_videos=False)
                self.assertFalse(report["valid"])
                self.assertTrue(any("no verified physical change" in error for error in report["errors"]))

    def test_mismatched_or_missing_snapshot_cannot_authorize_confirmation(self):
        path, rows, archive, snapshot = fixture_snapshots(self.root)
        rows[1].update(control_mode="GRASP_CONFIRM", capture_reason="physical_change")
        write_jsonl(path, rows)
        snapshot["qvel"][1, 18] = .04
        for corruption in ("time", "state", "action", "qpos", "missing_frame", "missing_field", "wrong_hash"):
            with self.subTest(corruption=corruption):
                block = {key: values.copy() for key, values in snapshot.items()}
                write_json(archive.parent / "task3.json", {"config_hash": "fixture-config"})
                if corruption == "time":
                    block["time"][1] += .001
                elif corruption == "state":
                    block["state"][1, 0] = .1
                elif corruption == "action":
                    block["action"][1, 0] = .2
                elif corruption == "qpos":
                    block["qpos"][1, 0] = .1
                elif corruption == "missing_frame":
                    block = {key: values[:2] for key, values in block.items()}
                elif corruption == "missing_field":
                    block.pop("qvel")
                else:
                    write_json(archive.parent / "task3.json", {"config_hash": "other-episode"})
                np.savez_compressed(archive, **block)
                report = validate_dataset(self.root, check_videos=False)
                self.assertFalse(report["valid"], corruption)
                self.assertTrue(any("snapshot" in error for error in report["errors"]))

    def test_wrong_snapshot_block_index_cannot_shift_frame_mapping(self):
        path, rows, archive, snapshot = fixture_snapshots(self.root)
        rows[1].update(control_mode="GRASP_CONFIRM", capture_reason="physical_change")
        write_jsonl(path, rows)
        snapshot["qvel"][1, 18] = .04
        np.savez_compressed(archive, **snapshot)
        archive.rename(archive.with_name("snapshots_000001.npz"))
        report = validate_dataset(self.root, check_videos=False)
        self.assertFalse(report["valid"])
        self.assertTrue(any("snapshot block ordering/gap" in error for error in report["errors"]))

    def test_legacy_motion_cadence_warns_without_rejecting_or_retiming_data(self):
        path = self.root / "research/episode_000000.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        shard = self.root / "data/chunk-000/file-000.parquet"
        data = pq.read_table(shard).to_pydict()
        for i, row in enumerate(rows):
            row["action"][0] += i * .001
            data["action"][i] = row["action"]
        pq.write_table(pa.Table.from_pydict(data), shard)
        before = shard.read_bytes()
        for times, expected_warning in (([0, .05, .1], True), ([0, .034, .066], False)):
            with self.subTest(times=times):
                for row, value in zip(rows, times):
                    row["sim_time"] = value
                write_jsonl(path, rows)
                report = validate_dataset(self.root, check_videos=False)
                self.assertTrue(report["valid"], report["errors"])
                self.assertEqual(any("cadence compression" in warning for warning in report["warnings"]), expected_warning)
                self.assertEqual(shard.read_bytes(), before, "Auditing must never rewrite training samples")

    def test_training_statistics_exclude_validation_episode(self):
        stats = numeric_training_stats(self.root, [0])
        self.assertEqual(stats["observation.state"]["mean"][0], 0.)
        self.assertAlmostEqual(float(stats["action"]["mean"][0]), .1)
        self.assertEqual(stats["action"]["count"].tolist(), [3])

    def test_checkpoint_binds_schema_and_episode_holdout(self):
        train = SimpleNamespace(episodes=[0], num_episodes=1, meta=SimpleNamespace(stats={}))
        heldout = SimpleNamespace(episodes=[1], meta=SimpleNamespace(stats={}))
        calls = []
        fake_trainer = SimpleNamespace(make_train_eval_datasets=lambda cfg: (train, heldout),
                                       save_checkpoint=lambda **kwargs: calls.append(kwargs),
                                       update_last_checkpoint=lambda path: path)
        output = self.root / "checkpoint-test"
        provenance = {"task3": {"schema_version": "task3-v1"}}
        install_training_hooks(fake_trainer, self.root, output, provenance, {"valid": True})
        fake_trainer.make_train_eval_datasets(None)
        self.assertEqual(provenance["train_episode_indices"], [0])
        self.assertEqual(provenance["validation_episode_indices"], [1])
        self.assertEqual(heldout.meta.stats["observation.state"]["mean"][0], 0.)
        fake_trainer.save_checkpoint(checkpoint_dir=output / "checkpoints/000001")
        checkpoint_metadata = output / "checkpoints/000001/pretrained_model/task3_metadata.json"
        self.assertTrue(checkpoint_metadata.is_file())
        self.assertEqual(len(calls), 1)

    def test_teleop_uses_wall_time_and_does_not_invent_annotations(self):
        rows, report = evaluate_roots([self.root])
        self.assertEqual(report["success_rate"], 1.)
        self.assertEqual(report["mean_success_completion_time_s"], .5)
        self.assertIsNone(rows[0]["false_gripper_trigger_count"])

    def test_policy_filter_common_motion_orientation_and_atomic_gripper(self):
        previous = np.zeros(14)
        desired = np.ones(14)
        desired[6], desired[13] = 0., 255.
        action, details = safe_policy_action(desired, previous, {"task_state": "DUAL_GRASPED"})
        np.testing.assert_allclose(action[:3], action[7:10])
        self.assertLessEqual(np.linalg.norm(action[:3]), .008001)
        self.assertEqual(action[6], action[13])
        self.assertTrue(np.all(action[[3, 4, 5, 10, 11, 12]] == 0))
        self.assertTrue(details["cooperative_filter"])
        with self.assertRaises(ValueError):
            safe_policy_action([np.nan] * 14, previous, {})


if __name__ == "__main__":
    unittest.main()
