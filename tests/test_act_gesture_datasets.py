"""Regression coverage for ACT auxiliary dataset timestamp compatibility."""

import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lerobot.datasets import factory as dataset_factory

from training.ACT.train_act_variation_gesture import (
    GESTURE_KEY,
    HAND_DELTA_KEY,
    HAND_VALID_KEY,
    install_auxiliary,
)


class GestureDatasetCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.policy_cfg = SimpleNamespace(
            observation_delta_indices=[-1, 0],
            action_delta_indices=[-1, 0, 1, 2],
            reward_delta_indices=None,
        )
        self.metadata = SimpleNamespace(
            fps=30,
            features={key: {} for key in (
                "observation.state", "observation.images.overview", "action",
                GESTURE_KEY, HAND_DELTA_KEY, HAND_VALID_KEY,
            )},
        )

    def make_training_module(self, builder, *, hand_delta=False):
        train_module = SimpleNamespace(
            make_train_eval_datasets=builder,
            make_policy=lambda *args, **kwargs: None,
        )
        install_auxiliary(
            train_module, gesture_weight=0.1,
            hand_delta_weight=0.1 if hand_delta else 0.0,
        )
        return train_module

    def test_legacy_two_argument_resolver(self):
        original_offsets = {"action": [-1 / 30, 0, 1 / 30, 2 / 30]}

        def legacy_resolve(cfg, ds_meta):
            self.assertIs(cfg, self.policy_cfg)
            self.assertIs(ds_meta, self.metadata)
            return original_offsets

        def builder(_cfg):
            return dataset_factory.resolve_delta_timestamps(
                self.policy_cfg, self.metadata,
            )

        with patch.object(dataset_factory, "resolve_delta_timestamps", legacy_resolve):
            module = self.make_training_module(builder)
            result = module.make_train_eval_datasets(SimpleNamespace())
            self.assertEqual(result[GESTURE_KEY], original_offsets["action"])
            self.assertIsNot(result[GESTURE_KEY], original_offsets["action"])
            self.assertNotIn(GESTURE_KEY, original_offsets)
            self.assertIs(dataset_factory.resolve_delta_timestamps, legacy_resolve)

    def test_rename_map_forwarded_by_position_and_keyword(self):
        rename_map = {"robot.state": "observation.state"}

        def current_resolve(cfg, ds_meta, rename_map):
            self.assertIs(cfg, self.policy_cfg)
            self.assertIs(ds_meta, self.metadata)
            self.assertIs(rename_map, expected_rename_map)
            return {"action": [0, 1 / 30]}

        expected_rename_map = rename_map
        for use_keywords in (False, True):
            with self.subTest(use_keywords=use_keywords):
                def builder(_cfg):
                    if use_keywords:
                        return dataset_factory.resolve_delta_timestamps(
                            cfg=self.policy_cfg, ds_meta=self.metadata,
                            rename_map=rename_map,
                        )
                    return dataset_factory.resolve_delta_timestamps(
                        self.policy_cfg, self.metadata, rename_map,
                    )

                with patch.object(dataset_factory, "resolve_delta_timestamps", current_resolve):
                    module = self.make_training_module(builder)
                    result = module.make_train_eval_datasets(SimpleNamespace())
                    self.assertEqual(result[GESTURE_KEY], result["action"])
                    self.assertIs(dataset_factory.resolve_delta_timestamps, current_resolve)

    def test_fixed_window_targets_use_selected_keys_and_action_timestamps(self):
        delta_key = "observation.hand_delta_window_uv"
        valid_key = "observation.hand_delta_window_valid"
        del self.metadata.features[HAND_DELTA_KEY]
        del self.metadata.features[HAND_VALID_KEY]
        self.metadata.features.update({delta_key: {}, valid_key: {}})

        def builder(_cfg):
            return dataset_factory.resolve_delta_timestamps(self.policy_cfg, self.metadata)

        module = SimpleNamespace(make_train_eval_datasets=builder,
                                 make_policy=lambda *args, **kwargs: None)
        install_auxiliary(module, gesture_weight=0.1, hand_delta_weight=0.1,
                          hand_delta_key=delta_key, hand_valid_key=valid_key)
        offsets = module.make_train_eval_datasets(SimpleNamespace())
        self.assertEqual(offsets[delta_key], offsets["action"])
        self.assertEqual(offsets[valid_key], offsets["action"])
        self.assertNotIn(HAND_DELTA_KEY, offsets)

    def test_installed_resolver_aligns_train_and_eval_auxiliary_targets(self):
        installed_resolver = dataset_factory.resolve_delta_timestamps
        supports_rename_map = "rename_map" in inspect.signature(installed_resolver).parameters

        def builder(_cfg):
            datasets = []
            for split in ("train", "eval"):
                args = [self.policy_cfg, self.metadata]
                if supports_rename_map:
                    args.append({})
                offsets = dataset_factory.resolve_delta_timestamps(*args)
                datasets.append(SimpleNamespace(split=split, delta_timestamps=offsets))
            return tuple(datasets)

        module = self.make_training_module(builder, hand_delta=True)
        train, evaluation = module.make_train_eval_datasets(SimpleNamespace())
        for dataset in (train, evaluation):
            with self.subTest(split=dataset.split):
                offsets = dataset.delta_timestamps
                self.assertEqual(offsets["action"], [-1 / 30, 0, 1 / 30, 2 / 30])
                for key in (GESTURE_KEY, HAND_DELTA_KEY, HAND_VALID_KEY):
                    self.assertEqual(offsets[key], offsets["action"])
                self.assertEqual(offsets["observation.state"], [-1 / 30, 0])
        self.assertIs(dataset_factory.resolve_delta_timestamps, installed_resolver)

    def test_resolver_restored_when_dataset_building_fails(self):
        installed_resolver = dataset_factory.resolve_delta_timestamps

        def builder(_cfg):
            self.assertIsNot(dataset_factory.resolve_delta_timestamps, installed_resolver)
            raise RuntimeError("dataset construction failed")

        module = self.make_training_module(builder)
        with self.assertRaisesRegex(RuntimeError, "dataset construction failed"):
            module.make_train_eval_datasets(SimpleNamespace())
        self.assertIs(dataset_factory.resolve_delta_timestamps, installed_resolver)


if __name__ == "__main__":
    unittest.main()
