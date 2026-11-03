"""DP history, future labels, frozen scaling and official lightweight checks."""
import copy
import json
from pathlib import Path
import pickle
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dataset.task3 import training_pack
from tests.papera.test_safe_stitch_data import make_pack
from training.diffusion.data import DiffusionDataset, Normalizer, _fit_statistics, prepare_data, restore_data
from training.papera.pack_data import SafeStitchDataset, prepare_pack, resize_rgb


class DiffusionDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.pack = make_pack(Path(cls.temporary.name))
        cls.config = {"observation_horizon": 2, "prediction_horizon": 30,
                      "action_horizon": 8, "image_size": [64, 48]}
        cls.binding = prepare_data(cls.pack, "tray_transfer", cls.config, engineering=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def dataset(self, binding=None, split="train"):
        result = DiffusionDataset(self.binding if binding is None else binding, split=split)
        self.addCleanup(result.close)
        return result

    def copy_pack(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name)/"pack"
        shutil.copytree(self.pack, path)
        return path

    def test_sample_shapes_first_history_repeat_and_causal_next_tick(self):
        data = self.dataset()
        sample = data[0]
        self.assertEqual(sample["images"].shape, (2,1,3,48,64))
        self.assertEqual(sample["state"].shape, (2,16))
        self.assertEqual(sample["action"].shape, (30,14))
        self.assertEqual(sample["is_pad"].shape, (30,))
        self.assertEqual(sample["observation_raw_frame_index"].tolist(), [0,0])
        np.testing.assert_array_equal(sample["images"][0], sample["images"][1])
        arrays = data.load_episode(sample["episode_key"])
        np.testing.assert_allclose(data.normalizer.decode_action(sample["action"])[0],
                                   arrays["action"][0], atol=1e-7)
        self.assertEqual(sample["target_control_tick"][0], sample["source_control_tick"]+1)

    def test_defaults_use_one_observation_and_full_640_480_frame(self):
        binding = prepare_data(self.pack, "tray_transfer", {}, engineering=True)
        self.assertEqual(binding["config"]["observation_horizon"], 1)
        self.assertEqual(binding["config"]["image_size"], [640, 480])
        self.assertEqual(binding["config"]["prediction_horizon"], 30)
        sample = self.dataset(binding)[0]
        self.assertEqual(sample["images"].shape, (1,1,3,480,640))
        self.assertEqual(sample["state"].shape, (1,16))

    def test_act_full_frame_default_and_explicit_historical_image_size(self):
        default = prepare_pack(self.pack, checked=self.binding.checked)
        self.assertEqual(default["image_size"], [640,480])
        data = SafeStitchDataset(default)
        self.addCleanup(data.close)
        self.assertEqual(data[0]["images"].shape, (1,3,480,640))
        historical = prepare_pack(self.pack, checked=self.binding.checked, image_size=(320,240))
        legacy_data = SafeStitchDataset(historical)
        self.addCleanup(legacy_data.close)
        self.assertEqual(legacy_data[0]["images"].shape, (1,3,240,320))
        self.assertEqual(resize_rgb(np.zeros((48,64,3),np.uint8)).shape, (480,640,3))

    def test_history_and_future_use_only_certified_virtual_adjacency(self):
        data = self.dataset()
        sample = data[8]
        self.assertEqual(sample["observation_raw_frame_index"].tolist(), [7,8])
        self.assertEqual(sample["target_raw_frame_index"][:5].tolist(), [8,9,40,41,42])
        seam = data[10]
        self.assertEqual(seam["observation_raw_frame_index"].tolist(), [9,40])
        self.assertTrue(np.all(seam["observation_source_control_tick"] <= seam["source_control_tick"]))
        self.assertEqual(len(set(seam["observation_train_sequence_id"].tolist())), 1)

    def test_hard_boundary_repeats_first_observation_and_pads_targets(self):
        data = self.dataset()
        before = data[23]
        self.assertEqual(before["target_raw_frame_index"][:3].tolist(), [53,54,-1])
        self.assertEqual(int(before["is_pad"].sum()), 28)
        boundary = data[25]
        self.assertEqual(boundary["observation_raw_frame_index"].tolist(), [55,55])
        self.assertTrue((before["action"][before["is_pad"]] == 0).all())
        self.assertEqual(int(data[len(data)-1]["is_pad"].sum()), 29)

    def test_train_only_frozen_minmax_and_validation_extrapolation(self):
        train, val = self.dataset(), self.dataset(split="val")
        self.assertEqual(train.normalizer.stats, val.normalizer.stats)
        self.assertEqual(self.binding["stats"]["training_keys"], self.binding["train_keys"])
        self.assertEqual(self.binding["stats"]["state_min"][0], 0.)
        self.assertEqual(self.binding["stats"]["state_max"][0], 0.)
        self.assertGreater(float(val[0]["state"][0,0]), 1.)
        arrays = train.load_episode(train.keys[0])
        for name, encode, decode in (("state", train.normalizer.encode_state, train.normalizer.decode_state),
                                     ("action", train.normalizer.encode_action, train.normalizer.decode_action)):
            encoded = encode(arrays[name])
            self.assertLessEqual(float(np.abs(encoded).max()), 1.+1e-6)
            np.testing.assert_allclose(decode(encoded), arrays[name], atol=1e-7)
        self.assertFalse(train.normalizer.action_scale.flags.writeable)

    def test_circular_rpy_uses_short_branch_and_roundtrips(self):
        state = np.stack([np.zeros(16), np.ones(16)])
        action = np.zeros((2,14))
        action[:,3] = [np.pi-.02, -np.pi+.02]
        stats = _fit_statistics({"train": {"state": state, "action": action,
                                "action_mask": np.ones(2, bool)}}, ["train"])
        normalizer = Normalizer(stats)
        self.assertAlmostEqual(stats["action_max"][3]-stats["action_min"][3], .04)
        encoded = normalizer.encode_action(action)
        self.assertLessEqual(float(np.abs(encoded).max()), 1.+1e-6)
        np.testing.assert_allclose(normalizer.decode_action(encoded), action, atol=1e-6)
        np.testing.assert_allclose(normalizer.encode_state(state)[:,0], [-1.,1.])

    def test_json_resume_keeps_statistics_without_refitting(self):
        binding = json.loads(json.dumps(self.binding))
        with patch("training.diffusion.data._fit_statistics", side_effect=AssertionError("refit")), \
             patch.object(training_pack, "verify", wraps=training_pack.verify) as verify:
            live = restore_data(binding)
            data = self.dataset(live)
            self.dataset(live, split="val")
            self.assertEqual(verify.call_count, 1)
            self.assertEqual(dict(live), binding)
            np.testing.assert_array_equal(data[8]["images"], self.dataset()[8]["images"])
        changed = copy.deepcopy(binding)
        changed["stats"]["state_min"][0] += 1
        with self.assertRaisesRegex(ValueError, "binding was modified"):
            DiffusionDataset(changed)

    def test_explicit_task_engineering_and_split_are_required(self):
        with self.assertRaisesRegex(ValueError, "Cross-task"):
            prepare_data(self.pack, "tray_to_platform", self.config, engineering=True)
        with self.assertRaisesRegex(ValueError, "Engineering Pack"):
            prepare_data(self.pack, "tray_transfer", self.config)
        with self.assertRaisesRegex(ValueError, "frozen train split"):
            prepare_data(self.pack, "tray_transfer", self.config, engineering=True,
                         selected_train_keys=self.binding["validation_keys"])
        with self.assertRaises(FileNotFoundError):
            prepare_data(self.pack/"missing", "tray_transfer", self.config, engineering=True)

    def test_lightweight_validation_skips_file_and_current_source_hashes(self):
        with patch.object(training_pack, "sha256", side_effect=AssertionError("file hash")), \
             patch("tasks.registry.get", side_effect=AssertionError("current task source gate")):
            binding = prepare_data(self.pack, "tray_transfer", self.config, engineering=True)
            data = self.dataset(binding)
            self.assertEqual(data[0]["action"].shape, (30,14))
        self.assertIn("no_file_hashes", binding["verification"])

    def test_hash_audit_is_optional_and_default_official_verify_remains_strict(self):
        path = self.copy_pack()
        manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
        manifest["episodes"][0]["files"][0]["sha256"] = "0"*64
        training_pack.write_json(path/"manifest.json", manifest)
        self.assertTrue(training_pack.verify(path, check_hashes=False)["valid"])
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            training_pack.verify(path)

    def test_lightweight_checks_file_sizes_and_original_causal_evidence(self):
        path = self.copy_pack()
        manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
        entry = manifest["episodes"][0]
        arrays_path = path/entry["arrays_path"]
        with arrays_path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Cache file changed"):
            training_pack.verify(path, check_hashes=False)
        with np.load(arrays_path, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        arrays["action"][0,0] += .1
        np.savez_compressed(arrays_path, **arrays)
        for record in entry["files"]:
            if record["path"] == entry["arrays_path"]:
                record["size"] = arrays_path.stat().st_size
        training_pack.write_json(path/"manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "original Prepared relation: action"):
            training_pack.verify(path, check_hashes=False)

    def test_lightweight_rejects_mixed_frozen_task_and_broken_stitch_certificate(self):
        path = self.copy_pack()
        prepared_path = path/"prepared.manifest.json"
        frozen = json.loads(prepared_path.read_text(encoding="utf-8"))
        frozen["task_binding"]["task_version"] = "9.9.9"
        training_pack.write_json(prepared_path, frozen)
        manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
        for record in manifest["files"]:
            if record["path"] == "prepared.manifest.json":
                record["size"] = prepared_path.stat().st_size
        training_pack.write_json(path/"manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "task or engineering provenance mismatch"):
            training_pack.verify(path, check_hashes=False)
        frozen["task_binding"] = copy.deepcopy(manifest["task_binding"])
        training_pack.write_json(prepared_path, frozen)
        entry = manifest["episodes"][0]
        diagnostics_path = path/entry["diagnostics_path"]
        diagnostic = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        diagnostic["virtual_sequence"]["gaps"] = []
        training_pack.write_json(diagnostics_path, diagnostic)
        for record in manifest["files"]:
            if record["path"] == "prepared.manifest.json":
                record["size"] = prepared_path.stat().st_size
        for record in entry["files"]:
            if record["path"] == entry["diagnostics_path"]:
                record["size"] = diagnostics_path.stat().st_size
        training_pack.write_json(path/"manifest.json", manifest)
        with self.assertRaises(ValueError):
            training_pack.verify(path, check_hashes=False)

    def test_pickle_and_windows_spawn_workers_reopen_rgb_memmaps(self):
        import torch
        from torch.utils.data import DataLoader
        data = self.dataset()
        data[0]
        restored = pickle.loads(pickle.dumps(data))
        self.addCleanup(restored.close)
        self.assertEqual(restored.episodes, {})
        self.assertEqual(restored.images, {})
        np.testing.assert_array_equal(restored[0]["images"], data[0]["images"])
        self.assertIsInstance(restored.images[restored.keys[0]][0], np.memmap)
        expected = next(iter(DataLoader(data, batch_size=4, num_workers=0)))
        loader = DataLoader(data, batch_size=4, num_workers=2, multiprocessing_context="spawn")
        iterator = iter(loader)
        try:
            actual = next(iterator)
            for name in ("images", "state", "action", "is_pad", "observation_raw_frame_index"):
                torch.testing.assert_close(actual[name], expected[name])
        finally:
            iterator._shutdown_workers()


if __name__ == "__main__":
    unittest.main()
