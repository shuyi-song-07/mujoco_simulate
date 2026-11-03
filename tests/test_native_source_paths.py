"""Native discovery/identity fixtures, not substitutes for a full Raw audit.

Only tiny metadata and episode indexes are needed at this public input seam.
These tests neither decode videos nor claim the fixture is training-ready.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from collection.config import recording_config
from dataset.recorded_collection import process_recorded, recorded_groups, select_group
from dataset.task3.curation_io import discover_episodes
from tasks.artifacts import source_contract
from tasks.recorded_collection import recorded_collection_scope
from tasks.registry import ROOT, get


class NativeSourcePathTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="native_source_paths_")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.task = get("tray_transfer")
        self.binding = self.task.binding()
        self.config = recording_config(self.task)
        self.calls = []

        def process(*, source, **_):
            source_contract(discover_episodes(source))
            self.calls.append(Path(source).resolve())
            return [source]

        self.workflow = SimpleNamespace(root=self.root, source=self.root / "unused_default",
            task=self.task, engineering=False, process=process, _remember=lambda outputs: None)

    def native_index(self, parent, name, operator, binding=None, *, episode=0):
        dataset = Path(parent) / name
        episode_name = f"episode_{episode:06d}"
        (dataset / "meta/episodes").mkdir(parents=True)
        (dataset / f"research/{episode_name}_snapshots").mkdir(parents=True)
        binding = copy.deepcopy(self.binding if binding is None else binding)
        config = copy.deepcopy(self.config)
        config["task_binding"] = binding
        metadata = {"schema_version": "task3-papera-v2", "task_id": self.task.id,
            "task_binding": binding, "scene_id": binding["scene_id"], "session": name,
            "operator_id": operator, "outcome": "success", "input_kind": "human_teleop",
            "config": config, "config_hash": self.config_hash(config)}
        self.write_json(dataset / "meta/info.json", {"fps": 30, "total_episodes": 1,
            "features": {"observation.images.oblique": {"dtype": "video", "shape": [480, 640, 3]}}})
        pq.write_table(pa.Table.from_pylist([{"episode_index": episode}]),
                       dataset / "meta/episodes/index.parquet")
        (dataset / "research/episodes.jsonl").write_text(
            json.dumps({"episode_index": episode, "outcome": "success", "session": name}) + "\n",
            encoding="utf-8")
        (dataset / f"research/{episode_name}.jsonl").write_text("{}\n", encoding="utf-8")
        self.write_json(dataset / f"research/{episode_name}_snapshots/episode_metadata.json", metadata)
        return dataset

    @staticmethod
    def config_hash(config):
        return hashlib.sha256(json.dumps(config, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def write_json(path, value):
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    def test_relocated_root_and_different_operators_use_actual_native_discovery(self):
        original = self.root / "original_import"
        self.native_index(original, "session_alpha", "operator_alpha")
        self.native_index(original, "session_beta", "operator_beta")
        before = {path.relative_to(original): path.read_bytes()
                  for path in original.rglob("episode_metadata.json")}
        relocated = self.root / "another_directory" / "incoming_native"
        relocated.parent.mkdir()
        self.assertTrue(original.resolve().is_relative_to(self.root))
        self.assertTrue(relocated.resolve().is_relative_to(self.root))
        original.rename(relocated)
        groups = recorded_groups(self.workflow, source=relocated)
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0]["current"])
        self.assertEqual(len(groups[0]["refs"]), 2)
        self.assertEqual(source_contract(groups[0]["refs"])["task_binding"], self.binding)
        for ref in groups[0]["refs"]:
            self.assertTrue(Path(ref["metadata_path"]).is_relative_to(relocated))
        for relative, content in before.items():
            self.assertEqual((relocated / relative).read_bytes(), content)

    def test_missing_native_research_index_is_rejected(self):
        dataset = self.native_index(self.root / "incoming", "session_alpha", "operator_alpha")
        (dataset / "research/episodes.jsonl").unlink()
        with self.assertRaises(FileNotFoundError):
            discover_episodes(dataset)

    def test_relocation_does_not_authorize_different_camera_or_physics(self):
        source = self.root / "incoming"
        self.native_index(source, "session_alpha", "operator_alpha")
        dataset = self.native_index(source, "session_beta", "operator_beta")
        path = dataset / "research/episode_000000_snapshots/episode_metadata.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for field in ("camera", "physics"):
            changed = copy.deepcopy(original)
            if field == "camera":
                changed["config"]["recording"]["cameras"] = ["front"]
            else:
                changed["config"]["scene"]["table_top_z"] += .01
            changed["config_hash"] = self.config_hash(changed["config"])
            self.write_json(path, changed)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Unapproved frozen setting"):
                recorded_groups(self.workflow, source=source)

    def test_same_session_basename_from_distinct_roots_is_explicitly_rejected(self):
        first = self.native_index(self.root / "operator_alpha", "same_session", "operator_alpha")
        second = self.native_index(self.root / "operator_beta", "same_session", "operator_beta")
        refs = discover_episodes(first) + discover_episodes(second)
        with self.assertRaisesRegex(ValueError, "Duplicate|collision|basename|same.*session"):
            recorded_groups(self.workflow, refs=refs)

    def test_distinct_episode_keys_do_not_hide_a_session_basename_collision(self):
        first = self.native_index(self.root / "operator_alpha", "same_session", "operator_alpha")
        second = self.native_index(self.root / "operator_beta", "same_session", "operator_beta", episode=1)
        refs = discover_episodes(first) + discover_episodes(second)
        with self.assertRaisesRegex(ValueError, "Duplicate session basename"):
            recorded_groups(self.workflow, refs=refs)

    def equivalent_group(self, *, include_current=True):
        source = self.root / "incoming"
        if include_current:
            self.native_index(source, "current_session", "operator_alpha")
        collection = json.loads((ROOT / "config/collection.json").read_text(encoding="utf-8"))
        alternative = (json.dumps(collection, sort_keys=True, indent=1) + "\r\n").encode("utf-8")
        recorded = copy.deepcopy(self.binding)
        recorded["sources"]["config/collection.json"] = hashlib.sha256(alternative).hexdigest()
        self.assertNotEqual(recorded, self.binding)
        self.native_index(source, "relocated_session", "operator_beta", recorded)
        group = next(group for group in recorded_groups(self.workflow, source=source) if not group["current"])
        self.assertEqual(group["provenance"]["config_differences"], [])
        return recorded, group

    def test_collection_byte_hash_change_with_equal_frozen_values_remains_readable(self):
        recorded, group = self.equivalent_group()
        with recorded_collection_scope(group["binding"], group["provenance"]):
            self.assertEqual(source_contract(group["refs"])["task_binding"], recorded)

    def test_equivalent_recorded_only_source_checks_current_recording_configuration(self):
        recorded, group = self.equivalent_group(include_current=False)
        with recorded_collection_scope(group["binding"], group["provenance"]):
            self.assertEqual(source_contract(group["refs"])["task_binding"], recorded)

    def test_empty_diff_proof_rejects_wrong_reference_or_recorded_binding(self):
        from tasks.recorded_collection import EQUIVALENCE_FIELD
        _, group = self.equivalent_group()
        for field in ("reference_collection_config", "frozen_collection_config"):
            proof = copy.deepcopy(group["provenance"])
            proof[EQUIVALENCE_FIELD][field]["task_binding"]["scene_id"] = "another_scene"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "binding differs"):
                with recorded_collection_scope(group["binding"], proof):
                    self.fail("wrong binding was accepted")

    def test_empty_diff_proof_rechecks_actual_metadata_config_and_binding(self):
        from tasks.recorded_collection import EQUIVALENCE_FIELD
        _, group = self.equivalent_group()
        paths = [group["provenance"]["metadata_paths"][0],
                 group["provenance"][EQUIVALENCE_FIELD]["reference_metadata_path"]]
        for path in paths:
            original = Path(path).read_text(encoding="utf-8")
            for field in ("config", "binding"):
                metadata = json.loads(original)
                if field == "config":
                    metadata["config"]["scene"]["table_top_z"] += .01
                    metadata["config_hash"] = self.config_hash(metadata["config"])
                else:
                    metadata["task_binding"]["scene_id"] = "another_scene"
                self.write_json(path, metadata)
                with self.subTest(path=path, field=field), self.assertRaisesRegex(ValueError, "metadata.*differs"):
                    with recorded_collection_scope(group["binding"], group["provenance"]):
                        self.fail("changed metadata was accepted")
                Path(path).write_text(original, encoding="utf-8")

    def test_empty_diff_proof_cannot_authorize_two_invented_equal_configurations(self):
        from tasks.recorded_collection import EQUIVALENCE_FIELD
        _, group = self.equivalent_group()
        proof = copy.deepcopy(group["provenance"])
        for field in ("reference_collection_config", "frozen_collection_config"):
            proof[EQUIVALENCE_FIELD][field]["scene"]["table_top_z"] += .01
        with self.assertRaisesRegex(ValueError, "metadata.*differs"):
            with recorded_collection_scope(group["binding"], proof):
                self.fail("invented equal snapshots were accepted")

    def test_repeated_identical_inputs_reuse_the_selected_source_directory(self):
        source = self.root / "incoming"
        self.native_index(source, "session_alpha", "operator_alpha")
        process_recorded(self.workflow, source=source)
        process_recorded(self.workflow, source=source)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[0], self.calls[1])

    def test_reused_selection_rejects_an_extra_entry_without_removing_it(self):
        source = self.root / "incoming"
        self.native_index(source, "session_alpha", "operator_alpha")
        process_recorded(self.workflow, source=source)
        unexpected = self.calls[0] / "unexpected.txt"
        unexpected.write_text("preserve this diagnostic", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "directory entries differ"):
            process_recorded(self.workflow, source=source)
        self.assertEqual(unexpected.read_text(encoding="utf-8"), "preserve this diagnostic")
        self.assertEqual(len(self.calls), 1)

    def test_reused_selection_rejects_retargeted_alias_without_repairing_it(self):
        source = self.root / "incoming"
        self.native_index(source, "session_alpha", "operator_alpha")
        process_recorded(self.workflow, source=source)
        link = self.calls[0] / "session_alpha"
        target = self.native_index(self.root / "another_import", "session_alpha", "operator_beta")
        replacement = select_group({"sessions": {"session_alpha": str(target)}}, self.root / "replacement")
        substitute = replacement / "session_alpha"
        self.assertTrue(link.resolve().is_relative_to(self.root))
        self.assertTrue(substitute.resolve().is_relative_to(self.root))
        if link.is_symlink():
            link.unlink()
        else:
            link.rmdir()  # Remove only the Windows junction, preserving its target.
        self.assertTrue(link.resolve().is_relative_to(self.root))
        substitute.rename(link)
        with self.assertRaisesRegex(ValueError, "points to another source"):
            process_recorded(self.workflow, source=source)
        self.assertEqual(link.resolve(), target.resolve())
        self.assertEqual(len(self.calls), 1)


class EquivalentPreparedSourceTests(unittest.TestCase):
    def test_single_recorded_prepared_with_equal_configuration_can_be_merged(self):
        from unittest.mock import patch
        from dataset.task3.merge_prepared import merge_prepared, validate_manifest
        from tests.task3.merge_prepared_fixture import source_prepared_fixture
        with tempfile.TemporaryDirectory(prefix="equivalent_prepared_") as temporary:
            root = Path(temporary)
            with source_prepared_fixture(root / "sources") as inputs:
                source = inputs["roots"][0]
                manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
                binding = copy.deepcopy(inputs["canonical_binding"])
                binding["sources"]["config/collection.json"] = "f" * 64
                manifest["task_binding"] = binding
                for entry in manifest["episodes"]:
                    path = Path(entry["source_ref"]["metadata_path"])
                    metadata = json.loads(path.read_text(encoding="utf-8"))
                    metadata["task_binding"] = binding
                    metadata["config"]["task_binding"] = binding
                    path.write_text(json.dumps(metadata), encoding="utf-8")
                    fingerprint = next(item for item in entry["source_fingerprints"] if Path(item["path"]) == path)
                    fingerprint.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=path.stat().st_size)
                workflow = SimpleNamespace(source=root, task=get("tray_transfer"), engineering=True)
                with patch("collection.config.recording_config", return_value=inputs["reference_collection_config"]):
                    group = recorded_groups(workflow, refs=[entry["source_ref"] for entry in manifest["episodes"]])[0]
                    with recorded_collection_scope(binding, group["provenance"]) as proof:
                        manifest["recorded_collection_compatibility"] = proof
                    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                    merged = merge_prepared([source], root / "merged")
                self.assertEqual(merged["source_task_bindings"], [binding])
                self.assertEqual(len(merged["episodes"]), 2)
                self.assertEqual(validate_manifest(merged), merged)


if __name__ == "__main__":
    unittest.main()
