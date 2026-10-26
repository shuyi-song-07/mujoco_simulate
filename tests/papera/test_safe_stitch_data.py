"""Formal Pack -> standard ACT windows, causal provenance and worker reads."""
import copy
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

from dataset.task3 import prepare_paper_dataset as prepared
from dataset.task3 import training_pack
from tests.task3.test_sequence_stitching import observations, cut
from tests.task3.test_single_view_prepared_pack import fixture
from training.papera.pack_data import (
    SafeStitchDataset, prepare_pack, restore_binding, resize_rgb, virtual_runs,
)


def make_pack(root):
    run, manifest, split = fixture(root)
    for number, entry in enumerate(manifest["episodes"]):
        source = entry["video_sources"][0]
        count = 80
        media = Path(source["path"])
        with av.open(str(media), "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for i in range(count):
                image = av.VideoFrame.from_ndarray(np.full((48,64,3), i*3, np.uint8), format="rgb24")
                for packet in stream.encode(image):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        source.update(end_s=count/30, frame_count=count)
        entry["review_video_sources"] = [dict(source, camera=c) for c in ("top","front","side")]
        entry["source_fingerprints"] = [{"path": str(media), "size": media.stat().st_size,
            "mtime_ns": media.stat().st_mtime_ns, "sha256": training_pack.sha256(media)}]
        rows = cut(observations(count))
        for i, row in enumerate(rows):
            row.update(session=entry["session"], source_episode=entry["key"],
                       robot_phase="free_stationary", status_summary={"paused": False},
                       terminal={"actual_done": i == count-1, "success": i == count-1},
                       auxiliary={side: {"measurement_id": None, "category": None,
                           "category_mask": False, "geometry_mask": False,
                           "task_geometry_mask": False} for side in ("left", "right")},
                       policy_decision="keep" if row["keep"] else "remove")
            row["hard_boundary"] = i in (0,55)
            facts = [row["evidence"], row["evidence"]["next_control"],
                     *row["evidence"]["interval"]["controls"]]
            for item in facts:
                item["session"] = entry["session"]
                for side in ("left", "right"):
                    item["features"][side+".joints"][1] = [number*100.]*7
            for j in (*range(7), *range(8,15)):
                row["state_recorded"][j] = number*100.
            # Distinguish next-control action from the observation's held target.
            row["next_control_target"]["action"][0] += .0001
            row["evidence"]["next_control"]["features"]["left.target_position"][1][0] += .0001
        path = run/entry["index_path"]
        prepared.write_rows(path, rows)
        entry["index_sha256"] = training_pack.sha256(path)
        entry["metrics"]["raw_frames"] = count
    prepared.write_json(run/"manifest.json", manifest)
    with patch.object(prepared, "load_episode", return_value={"load_errors": [], "measurements": {}}):
        prepared.prepare(run, root/"prepared", review_mode="auto")
    training_pack.build(root/"prepared", root/"pack", split_file=split, width=64, height=48)
    return root/"pack"


class SafeStitchTrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.pack = make_pack(Path(cls.temporary.name))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.checked = training_pack.verify(self.pack)
        self.binding = prepare_pack(self.pack, checked=self.checked, image_size=(64,48))
        self.data = SafeStitchDataset(self.binding)
        self.addCleanup(self.data.close)

    def test_official_reader_shapes_and_actual_next_tick_action(self):
        sample = self.data[0]
        self.assertEqual(sample["images"].shape, (1,3,48,64))
        self.assertEqual(sample["state"].shape, (16,))
        self.assertEqual(sample["action"].shape, (30,14))
        arrays = self.data.load_episode(sample["episode_key"])
        np.testing.assert_allclose(self.data.normalizer.decode_action(sample["action"])[0],
                                   arrays["action"][0], atol=1e-7)
        self.assertEqual(sample["target_control_tick"][0], sample["source_control_tick"]+1)
        self.assertAlmostEqual(float(arrays["action"][0,0]), .0001, places=7)

    def test_safe_gap_is_crossed_with_raw_and_virtual_clocks_separate(self):
        sample = self.data[8]
        self.assertEqual(sample["target_raw_frame_index"][:5].tolist(), [8,9,40,41,42])
        self.assertEqual(sample["target_control_tick"][:5].tolist(), [17,19,81,83,85])
        self.assertAlmostEqual(sample["target_training_time_s"][2]-sample["target_training_time_s"][1], 1/30)
        self.assertAlmostEqual(sample["target_original_time_s"][2]-sample["target_original_time_s"][1], 31/30)
        self.assertGreater(self.binding["counts"]["train"]["safe_stitch_windows"], 0)

    def test_hard_boundary_and_episode_end_pad_without_crossing(self):
        sample = self.data[23]  # raw53, raw54 precede source hard boundary raw55.
        self.assertEqual(sample["target_raw_frame_index"][:3].tolist(), [53,54,-1])
        self.assertEqual(sample["is_pad"].sum(), 28)
        self.assertTrue((sample["action"][sample["is_pad"]] == 0).all())
        last = self.data[len(self.data)-1]
        self.assertEqual(last["is_pad"].sum(), 29)
        self.assertFalse(last["truncated"])

    def test_manual_exclusion_is_hard_even_if_physical_seam_is_safe(self):
        episode = training_pack.read_episode(self.pack, self.data.keys[0], checked=self.checked,
                                             include_evidence=False)
        self.addCleanup(episode["images"]["oblique"]._mmap.close)
        sequence = copy.deepcopy(episode["diagnostics"]["virtual_sequence"])
        sequence["gaps"][0]["decision_origin"] = "manual"
        runs, safe, hard = virtual_runs(episode["arrays"], sequence)
        self.assertEqual(runs[0], (0,10))
        self.assertEqual(safe, [])
        self.assertIn(10, hard)

    def test_missing_or_inconsistent_certificate_rejected_not_guessed_safe(self):
        episode = training_pack.read_episode(self.pack, self.data.keys[0], checked=self.checked,
                                             include_evidence=False)
        self.addCleanup(episode["images"]["oblique"]._mmap.close)
        sequence = copy.deepcopy(episode["diagnostics"]["virtual_sequence"])
        sequence["gaps"] = []
        with self.assertRaisesRegex(ValueError, "no Dataset seam"):
            virtual_runs(episode["arrays"], sequence)
        sequence = copy.deepcopy(episode["diagnostics"]["virtual_sequence"])
        sequence["gaps"][0]["kind"] = "hard_boundary"
        with self.assertRaisesRegex(ValueError, "Unsafe seam"):
            virtual_runs(episode["arrays"], sequence)

    def test_train_only_statistics_and_normalization_roundtrip(self):
        self.assertAlmostEqual(self.binding["stats"]["state_mean"][0], 0.)
        self.assertEqual(self.binding["stats"]["training_keys"], self.binding["train_keys"])
        val = SafeStitchDataset(self.binding, split="val")
        self.addCleanup(val.close)
        self.assertIs(val.normalizer.stats, self.data.normalizer.stats)
        self.assertGreater(float(val[0]["state"][0]), 100.)
        arrays = self.data.load_episode(self.data.keys[0])
        encoded = self.data.normalizer.encode_action(arrays["action"])
        np.testing.assert_allclose(self.data.normalizer.decode_action(encoded), arrays["action"], atol=1e-7)

    def test_selected_training_keys_cannot_include_validation(self):
        with self.assertRaisesRegex(ValueError, "frozen train split"):
            prepare_pack(self.pack, selected_train_keys=self.binding["validation_keys"], checked=self.checked)
        with self.assertRaisesRegex(ValueError, "crosses the frozen split"):
            SafeStitchDataset(self.binding, selected_keys=self.binding["validation_keys"])

    def test_frozen_binding_reverify_rejects_changed_stats_or_shape(self):
        restored = restore_binding(json.loads(json.dumps(self.binding)))
        self.assertEqual(dict(restored), dict(self.binding))
        changed = json.loads(json.dumps(self.binding))
        changed["stats"]["state_mean"][0] += 1
        with self.assertRaisesRegex(ValueError, "Frozen ACT data binding changed"):
            restore_binding(changed)
        with self.assertRaisesRegex(ValueError, "Image shape"):
            SafeStitchDataset(self.binding, image_size=(320,240))

    def test_plain_json_verify_report_cannot_bypass_full_verification(self):
        with self.assertRaisesRegex(ValueError, "live report"):
            training_pack.read_episode(self.pack, self.data.keys[0], checked=dict(self.checked),
                                       include_evidence=False)
        with self.assertRaisesRegex(ValueError, "requires"):
            training_pack.read_episode(self.pack, self.data.keys[0], include_evidence=False)

    def test_mutated_live_report_rejected(self):
        checked = copy.deepcopy(self.checked)
        checked["manifest"]["frame_count"] += 1
        with self.assertRaisesRegex(ValueError, "report was modified"):
            training_pack.read_episode(self.pack, self.data.keys[0], checked=checked,
                                       include_evidence=False)

    def test_file_change_since_verification_rejected(self):
        record = self.checked["manifest"]["episodes"][0]
        path = self.pack/record["arrays_path"]
        original = path.stat()
        try:
            os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns+10000000))
            with self.assertRaisesRegex(ValueError, "changed since full verification"):
                training_pack.read_episode(self.pack, self.data.keys[0], checked=self.checked,
                                           include_evidence=False)
        finally:
            os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))

    def test_dataset_pickle_omits_arrays_images_and_raw_evidence(self):
        self.data[0]
        self.assertTrue(self.data.images)
        state = self.data.__getstate__()
        self.assertEqual(state["images"], {})
        self.assertEqual(state["episodes"], {})
        restored = pickle.loads(pickle.dumps(self.data))
        self.addCleanup(restored.close)
        np.testing.assert_array_equal(restored[0]["images"], self.data[0]["images"])
        self.assertIsInstance(restored.images[restored.keys[0]][0], np.memmap)

    def test_windows_spawn_workers_reopen_official_rgb_memmaps(self):
        import torch
        from torch.utils.data import DataLoader
        expected = next(iter(DataLoader(self.data, batch_size=4, num_workers=0)))
        loader = DataLoader(self.data, batch_size=4, num_workers=2,
                            multiprocessing_context="spawn", persistent_workers=False)
        iterator = iter(loader)
        try:
            actual = next(iterator)
            torch.testing.assert_close(actual["images"], expected["images"])
            torch.testing.assert_close(actual["action"], expected["action"])
            torch.testing.assert_close(actual["target_control_tick"], expected["target_control_tick"])
        finally:
            iterator._shutdown_workers()

    def test_shared_resize_is_whole_frame_without_crop(self):
        image = np.arange(48*64*3, dtype=np.uint8).reshape(48,64,3)
        np.testing.assert_array_equal(resize_rgb(image, (40,32)),
            av.VideoFrame.from_ndarray(image,format="rgb24").reformat(
                width=40,height=32,format="rgb24").to_ndarray())


if __name__ == "__main__":
    unittest.main()
