"""ACT excludes EMA from the CLI and checkpoints without altering DP's schema."""

from dataclasses import fields
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import draccus
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

from training.ACT.act_training_config import ACTTrainPipelineConfig, install_act_training_config
from training.ACT.train_act_variation_baseline import build_cli, make_parser


class ACTTrainingConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The real training module imports ACT before decoding the CLI.
        from lerobot.policies.act.configuration_act import ACTConfig
        cls.policy_config_class = ACTConfig

    def test_all_modes_validate_and_save_without_ema(self):
        for auxiliary, motion in ((False, False), (True, False), (True, True)):
            with self.subTest(auxiliary=auxiliary, motion=motion), tempfile.TemporaryDirectory() as root:
                args = make_parser(auxiliary=auxiliary, motion=motion).parse_args([
                    '--dataset-root', root, '--device', 'cpu',
                    '--output-dir', str(Path(root) / 'output'),
                ])
                cli = build_cli(args, auxiliary=auxiliary)
                with patch.object(sys, 'argv', ['train_act.py', *cli]):
                    config = draccus.parse(ACTTrainPipelineConfig, args=cli)
                    config.validate()
                self.assertNotIn('ema', {field.name for field in fields(config)})
                self.assertNotIn('ema', config.to_dict())
                self.assertFalse(config.ema.enable)
                self.assertFalse(config.ema.use_for_eval)
                config.save_pretrained(root)
                payload = json.loads((Path(root) / 'train_config.json').read_text())
                self.assertNotIn('ema', payload)
                # The ordinary LeRobot loader can still read this ACT checkpoint.
                with patch.object(sys, 'argv', ['test']):
                    loaded = TrainPipelineConfig.from_pretrained(root)
                self.assertEqual(loaded.policy.type, 'act')
                if hasattr(loaded, 'ema'):
                    self.assertFalse(loaded.ema.enable)

    def test_installed_entrypoint_uses_act_schema(self):
        seen = []

        @parser.wrap()
        def train(cfg: TrainPipelineConfig):
            seen.append(cfg)

        module = SimpleNamespace(train=train)
        install_act_training_config(module)
        with patch.object(sys, 'argv', ['train_act.py', '--dataset.repo_id=local/test', '--policy.type=act']):
            module.train()
        self.assertIs(type(seen[0]), ACTTrainPipelineConfig)
        self.assertNotIn('ema', seen[0].to_dict())

    def test_shared_schema_still_has_ema_when_available(self):
        shared_fields = {field.name for field in fields(TrainPipelineConfig)}
        if 'ema' in shared_fields:
            config = draccus.parse(TrainPipelineConfig, args=[
                '--dataset.repo_id=local/test', '--policy.type=act', '--ema.enable=true',
            ])
            self.assertTrue(config.ema.enable)
            self.assertIn('ema', config.to_dict())

    def test_act_cli_rejects_ema_options(self):
        with patch.object(sys, 'stderr', StringIO()), self.assertRaises(SystemExit):
            draccus.parse(ACTTrainPipelineConfig, args=[
                '--dataset.repo_id=local/test', '--policy.type=act', '--ema.enable=true',
            ])


if __name__ == '__main__':
    unittest.main()
