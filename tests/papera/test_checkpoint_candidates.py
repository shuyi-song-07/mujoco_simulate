"""Checkpoint retention must not change training or historical resume contracts."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from training.papera.source_compatibility import (
    REVIEWED_TRAINER_PAIRS, compatible_resume_provenance, verify_training_sources,
)
from training.papera.train import (
    atomic_checkpoint, candidate_due, make_parser, retain_candidate,
    rng_state, seed_everything, validate_config, validate_extension_config,
    load_extension_source,
)


ROOT = Path(__file__).resolve().parents[2]


class CheckpointCandidateTests(unittest.TestCase):
    def test_disabled_old_config_stays_unmodified_and_performs_no_io(self):
        config = {"steps": 100}
        before = copy.deepcopy(config)
        calls = []
        self.assertFalse(candidate_due(config, 100))
        self.assertIsNone(retain_candidate("unused", calls.append, config, 100, "run"))
        self.assertEqual(config, before)
        self.assertEqual(calls, [])
        self.assertFalse(candidate_due({**config, "candidate_save_every": 0}, 100))

    def test_schedule_includes_final_step_without_scheduling_validation(self):
        config = {"steps": 11, "candidate_save_every": 4}
        self.assertEqual([s for s in range(1, 12) if candidate_due(config, s)], [4, 8, 11])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "checkpoints").mkdir()
            payload = {"step": 4, "model": {"w": torch.tensor([2.])},
                       "optimizer": {"state": {0: {"exp_avg": torch.tensor([.1])}}},
                       "scheduler": {"last_epoch": 4}, "sampler": {"cursor": 8},
                       "rng": rng_state(), "loader_generator": torch.Generator().get_state()}
            save = lambda name: atomic_checkpoint(out / "checkpoints" / name, payload)
            entry = retain_candidate(out, save, config, 4, "run")
            self.assertIsNone(entry["validation"])
            self.assertIsNone(entry["validation_step"])
            self.assertIsNone(entry["validation_source"])
            restored = torch.load(out / "checkpoints/step_000004.pt", weights_only=True)
            self.assertEqual(set(restored), set(payload))
            torch.testing.assert_close(restored["model"]["w"], payload["model"]["w"])
            self.assertEqual(restored["scheduler"], payload["scheduler"])
            self.assertEqual(restored["sampler"], payload["sampler"])
            self.assertFalse((out / "checkpoints/best.pt").exists())
            self.assertFalse((out / "checkpoints/last.pt").exists())

    def test_current_validation_index_and_immutable_candidates(self):
        config = {"steps": 10, "candidate_save_every": 5}
        metrics = {"step": 5, "event": "validation", "prior_action_l1": .2,
                   "validation_scope": "heldout_sessions", "prior_only": True}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "checkpoints").mkdir()
            save = lambda name: atomic_checkpoint(out / "checkpoints" / name, {"step": 5})
            entry = retain_candidate(out, save, config, 5, "run", metrics)
            self.assertEqual(entry["saved_step"], entry["validation_step"])
            self.assertEqual(entry["validation_scope"], "heldout_sessions")
            self.assertEqual(entry["validation_source"], "metrics.jsonl:validation")
            self.assertEqual(entry["validation"], metrics)
            index = json.loads((out / "checkpoints/candidates.json").read_text(encoding="utf-8"))
            self.assertEqual(index["candidates"], [entry])
            with self.assertRaisesRegex(ValueError, "already exists"):
                retain_candidate(out, save, config, 5, "run", metrics)
            with self.assertRaisesRegex(ValueError, "saved step"):
                retain_candidate(out, save, config, 10, "run", metrics)
            with self.assertRaisesRegex(ValueError, "different run"):
                retain_candidate(out, save, config, 10, "other")

    def test_checkpoint_and_index_io_preserve_all_rng_streams(self):
        seed_everything(91)
        expected = (random.random(), np.random.rand(4), torch.rand(4))
        seed_everything(91)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "checkpoints").mkdir()
            save = lambda name: atomic_checkpoint(out / "checkpoints" / name, {"rng": rng_state()})
            retain_candidate(out, save, {"steps": 1, "candidate_save_every": 1}, 1, "run")
        actual = (random.random(), np.random.rand(4), torch.rand(4))
        self.assertEqual(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)

    def test_config_default_validation_and_cli(self):
        config = json.loads((ROOT / "config/training_baseline.json").read_text(encoding="utf-8"))
        self.assertEqual(config["candidate_save_every"], 5000)
        for value in (0, 1, 5000):
            validate_config({**config, "candidate_save_every": value})
        for value in (-1, True, 2.5, "5000", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "candidate_save_every"):
                validate_config({**config, "candidate_save_every": value})
        self.assertEqual(make_parser().parse_args(
            ["--task", "tray_to_platform", "--candidate-save-every", "2000"]).candidate_save_every, 2000)

    def test_extension_permits_only_retention_alongside_existing_step_lr_changes(self):
        original = {"steps": 40000, "learning_rate": .0001}
        source = {"run_doc": {"provenance": {"config": original}},
                  "checkpoint": {"step": 40000}, "start_lrs": [1e-5, 1e-6]}
        new = {**original, "steps": 45000, "candidate_save_every": 1000,
               "lr_schedule": {"kind": "constant_checkpoint_lr", "start_step": 40000,
                               "start_lrs": source["start_lrs"]},
               "_help": {"steps": "从已完成的40000步 last.pt 延续，累计目标45000步；"
                                  "续训阶段保持checkpoint中的最低学习率，不重启升高。"}}
        validate_extension_config(new, source, 45000)
        with self.assertRaisesRegex(ValueError, "Continuation"):
            validate_extension_config({**new, "learning_rate": .0002}, source, 45000)
        self.assertNotIn("candidate_save_every", original)


class TrainingSourceCompatibilityTests(unittest.TestCase):
    def provenance(self):
        files = {"model_source_sha256": "act_model.py", "reader_source_sha256": "pack_data.py",
                 "normalizer_source_sha256": "data.py", "trainer_source_sha256": "train.py",
                 "engine_source_sha256": "engine.py"}
        return {key: hashlib.sha256((ROOT / "training/papera" / file).read_bytes()).hexdigest()
                for key, file in files.items()}

    def test_exact_sources_and_only_reviewed_trainer_pair(self):
        current = self.provenance()
        self.assertEqual(verify_training_sources(current)["compatibility"], "exact-current-sources")
        # A synthetic interface-only migration tests the allowlist mechanism;
        # restoring the official network must not be added to that allowlist.
        old = "synthetic-interface-only-trainer"
        previous = {**current, "trainer_source_sha256": old}
        pair = (old, current['trainer_source_sha256'])
        with patch.dict(REVIEWED_TRAINER_PAIRS, {pair: 'synthetic-reviewed-interface-only-fixture'}):
            self.assertEqual(verify_training_sources(previous)["compatibility"], "reviewed-training-interface-only")
            self.assertEqual(compatible_resume_provenance(current, previous), previous)
            for field in current:
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Frozen training source changed"):
                    verify_training_sources({**previous, field: "not-reviewed"})

    def test_smoke_guard_interface_fixture_preserves_original_trainer_identity(self):
        current = self.provenance()
        for old in (
            'synthetic-trainer-before-retention',
            'synthetic-trainer-before-continuation',
            'synthetic-trainer-before-smoke-guard',
        ):
            pair = (old, current['trainer_source_sha256'])
            with self.subTest(fixture_trainer=old), patch.dict(REVIEWED_TRAINER_PAIRS,
                    {pair: 'synthetic-reviewed-smoke-interface-only-fixture'}):
                previous = {**current, 'trainer_source_sha256': old}
                report = verify_training_sources(previous)
                self.assertEqual(report['compatibility'], 'reviewed-training-interface-only')
                self.assertIn('smoke', ' '.join(report['reviewed_changes']))
                self.assertEqual(compatible_resume_provenance(current, previous), previous)

    def test_reviewed_source_migration_cannot_change_config_data_or_model(self):
        current = self.provenance()
        old = "synthetic-interface-only-trainer"
        previous = {**current, "trainer_source_sha256": old, "config": {"steps": 40000}, "pack": "A"}
        migrated = {**previous, "trainer_source_sha256": current["trainer_source_sha256"]}
        pair = (old, current['trainer_source_sha256'])
        with patch.dict(REVIEWED_TRAINER_PAIRS, {pair: 'synthetic-reviewed-interface-only-fixture'}):
            self.assertEqual(compatible_resume_provenance(migrated, previous), previous)
            for changed in ({"config": {"steps": 40001}}, {"pack": "B"},
                            {"config": {"steps": 40000, "candidate_save_every": 5000}},
                            {"model_source_sha256": "different-network"}):
                with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "Checkpoint provenance changed"):
                    compatible_resume_provenance({**migrated, **changed}, previous)

    def test_historical_trainer_pairs_do_not_authorize_official_core_migration(self):
        current = self.provenance()
        self.assertTrue(REVIEWED_TRAINER_PAIRS)
        self.assertTrue(all(new != current['trainer_source_sha256'] for _, new in REVIEWED_TRAINER_PAIRS))
        for old in {pair[0] for pair in REVIEWED_TRAINER_PAIRS}:
            previous = {**current, 'trainer_source_sha256': old}
            with self.subTest(historical_trainer=old):
                with self.assertRaisesRegex(ValueError, 'Frozen training source changed'):
                    verify_training_sources(previous)
                with self.assertRaisesRegex(ValueError, 'Checkpoint provenance changed'):
                    compatible_resume_provenance(current, previous)

    def test_legacy_checkpoint_contract_is_rejected_before_provenance_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'checkpoints').mkdir()
            for name in ('run', 'config', 'completion', 'pack_binding', 'split', 'stats'):
                (root/f'{name}.json').write_text('{}', encoding='utf-8')
            checkpoint = {'version': 'papera-act-offline-checkpoint-v1'}
            torch.save(checkpoint, root/'checkpoints/last.pt')
            torch.save(checkpoint, root/'checkpoints/best.pt')
            with self.assertRaisesRegex(ValueError, 'Legacy local ACT checkpoint'):
                load_extension_source(root/'checkpoints/last.pt', 60000)


if __name__ == "__main__":
    unittest.main()
