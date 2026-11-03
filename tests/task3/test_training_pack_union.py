"""One official Pack preserves, and validates, each Prepared source identity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dataset.task3 import prepare_paper_dataset as prepared
from dataset.task3 import training_pack as packer
from dataset.task3.merge_prepared import merge_prepared
from tests.task3.merge_prepared_fixture import source_prepared_fixture
from tests.task3.test_single_view_prepared_pack import fixture


class TrainingPackUnionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        run, _, self.split = fixture(self.root)
        self.prepared = self.root / "prepared"
        with patch.object(prepared, "load_episode", return_value={"load_errors": [], "measurements": {}}):
            self.manifest = prepared.prepare(run, self.prepared, review_mode="auto")

    def write_prepared(self):
        prepared.write_json(self.prepared / "manifest.json", self.manifest)

    def test_episode_source_cannot_be_silently_ignored_without_union_declaration(self):
        episode = self.manifest["episodes"][0]
        episode.update(source_task_binding=copy.deepcopy(self.manifest["task_binding"]),
                       source_prepared=str(self.root / "another_prepared"),
                       source_prepared_manifest_sha256="a" * 64)
        self.write_prepared()
        output = self.root / "pack"
        with self.assertRaisesRegex(ValueError, "source|provenance|union"):
            packer.build(self.prepared, output, split_file=self.split, width=64, height=48)
        self.assertFalse(output.exists())


class TrainingPackSourceLedgerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="training_pack_union_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        sources = source_prepared_fixture(self.root / "sources")
        self.inputs = sources.__enter__()
        self.addCleanup(sources.__exit__, None, None, None)
        self.prepared = self.root / "merged"
        self.manifest = merge_prepared(self.inputs["roots"], self.prepared,
                                       split_file=self.inputs["split"])
        self.pack = self.root / "pack"

    def build(self):
        return packer.build(self.prepared, self.pack, split_file=self.inputs["split"],
                            width=64, height=48)

    def test_one_pack_preserves_source_ledgers_and_uses_the_original_reader(self):
        manifest = self.build()
        frozen = json.loads((self.pack / "prepared.manifest.json").read_text(encoding="utf-8"))
        for name in packer.SOURCE_MANIFEST_FIELDS:
            self.assertEqual(manifest[name], self.manifest[name])
            self.assertEqual(frozen[name], self.manifest[name])
        self.assertEqual(manifest["task_binding"], self.inputs["canonical_binding"])
        self.assertEqual(manifest["episode_count"], 4)
        for check_hashes in (True, False):
            checked = packer.verify(self.pack, check_hashes=check_hashes)
            self.assertTrue(checked["valid"])
            self.assertEqual(checked["frames"], 24)
            for original in self.manifest["episodes"]:
                value = packer.read_episode(self.pack, original["key"], checked=checked,
                                            include_evidence=False)
                try:
                    for name in packer.SOURCE_EPISODE_FIELDS:
                        self.assertEqual(value["entry"][name], original[name])
                    self.assertEqual(value["images"]["oblique"].shape, (6, 48, 64, 3))
                    self.assertEqual(value["arrays"]["state"].shape, (6, 16))
                    self.assertEqual(value["arrays"]["action"].shape, (6, 14))
                finally:
                    value["images"]["oblique"]._mmap.close()

    def test_offline_verification_rejects_each_episode_source_association_change(self):
        manifest = self.build()
        for name, value in (("source_task_binding", self.inputs["historical_binding"]),
                            ("source_prepared", str(self.root / "unregistered_source")),
                            ("source_prepared_manifest_sha256", "f" * 64)):
            altered = copy.deepcopy(manifest)
            altered["episodes"][0][name] = value
            prepared.write_json(self.pack / "manifest.json", altered)
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, "source provenance"):
                packer.verify(self.pack, check_hashes=False)
            with self.subTest(direct_reader=name), self.assertRaisesRegex(ValueError, "source provenance"):
                packer.read_episode(self.pack, altered["episodes"][0]["key"], verify_files=False)

    def test_offline_verification_rejects_changed_or_missing_global_source_declarations(self):
        manifest = self.build()
        for name in packer.SOURCE_MANIFEST_FIELDS:
            altered = copy.deepcopy(manifest)
            altered.pop(name)
            prepared.write_json(self.pack / "manifest.json", altered)
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, "source provenance"):
                packer.verify(self.pack, check_hashes=False)

    def test_matching_outer_and_embedded_ledgers_cannot_mix_scene_schema_or_camera(self):
        manifest = self.build()
        frozen_path = self.pack / "prepared.manifest.json"
        original = json.loads(frozen_path.read_text(encoding="utf-8"))
        for field, value in (("scene_id", "another_scene"), ("schema", {}),
                             ("task_id", "another_task")):
            altered = copy.deepcopy(manifest)
            frozen = copy.deepcopy(original)
            frozen["source_task_bindings"][-1][field] = value
            altered["source_task_bindings"] = copy.deepcopy(frozen["source_task_bindings"])
            prepared.write_json(frozen_path, frozen)
            prepared.write_json(self.pack / "manifest.json", altered)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "bindings differ"):
                packer.verify(self.pack, check_hashes=False)
        altered, frozen = copy.deepcopy(manifest), copy.deepcopy(original)
        episode = frozen["episodes"][-1]
        views = episode.get("training_video_sources", episode["video_sources"])
        views[0]["camera"] = "front"
        altered["episodes"][-1]["source_video"]["camera"] = "front"
        prepared.write_json(frozen_path, frozen)
        prepared.write_json(self.pack / "manifest.json", altered)
        with self.assertRaisesRegex(ValueError, "source camera"):
            packer.verify(self.pack, check_hashes=False)

    def test_source_metadata_change_cannot_reassign_an_episode(self):
        manifest = self.build()
        altered = copy.deepcopy(manifest)
        altered["episodes"][0]["source_ref"] = altered["episodes"][1]["source_ref"]
        prepared.write_json(self.pack / "manifest.json", altered)
        with self.assertRaisesRegex(ValueError, "source evidence"):
            packer.verify(self.pack, check_hashes=False)

    def test_source_video_fingerprint_must_match_its_frozen_prepared_declaration(self):
        manifest = self.build()
        manifest["episodes"][0]["source_video"]["sha256"] = "f" * 64
        prepared.write_json(self.pack / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "source video fingerprint"):
            packer.verify(self.pack, check_hashes=False)

    def test_union_frozen_split_cannot_be_replaced_by_another_valid_session_split(self):
        source = json.loads(self.inputs["split"].read_text(encoding="utf-8"))
        replacement = self.root / "another_split.json"
        prepared.write_json(replacement, {"train_sessions": source["val_sessions"],
                                           "val_sessions": source["train_sessions"]})
        with self.assertRaisesRegex(ValueError, "frozen.*split|split.*frozen"):
            packer.build(self.prepared, self.pack, split_file=replacement, width=64, height=48)
        self.assertFalse(self.pack.exists())

    def test_pack_remains_self_contained_when_original_prepared_manifests_are_unavailable(self):
        self.build()
        for source in self.inputs["roots"]:
            (source / "manifest.json").unlink()
        self.assertTrue(packer.verify(self.pack, check_hashes=False)["valid"])
        self.assertTrue(packer.verify(self.pack, check_hashes=True)["valid"])


if __name__ == "__main__":
    unittest.main()
