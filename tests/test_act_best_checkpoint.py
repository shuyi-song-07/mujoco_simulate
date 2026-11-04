"""Exercise validation tracking with both LeRobot forward calling conventions."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from training.ACT.best_checkpoint import install_best_checkpoint
from training.ACT.train_act_variation_baseline import build_cli, make_parser


class LossPolicy(torch.nn.Module):
    def forward(self, loss):
        return torch.tensor(loss), {"metric": 1}


class BestCheckpointTests(unittest.TestCase):
    def test_best_tracks_full_precision_loss_and_keeps_regular_checkpoints(self):
        for direct_forward in (True, False):
            with self.subTest(direct_forward=direct_forward), tempfile.TemporaryDirectory() as root:
                def save_checkpoint(checkpoint_dir, step, cfg, policy, accelerator=None):
                    path = Path(checkpoint_dir) / "pretrained_model"
                    path.mkdir(parents=True)
                    (path / "model.safetensors").write_text(str(step))

                module = SimpleNamespace(make_policy=lambda: LossPolicy(), save_checkpoint=save_checkpoint)
                install_best_checkpoint(module)
                policy = module.make_policy()
                cfg = SimpleNamespace(eval_steps=2)
                for step, losses in ((2, [1.0, 3.0]), (4, [2.1, 2.2]),
                                     (6, [1.99999, 1.99999]), (8, [float("nan")])):
                    policy.train()
                    policy.forward(-100.0)  # Training loss must not affect selection.
                    policy.eval()
                    for loss in losses:
                        output = policy.forward(loss) if direct_forward else policy(loss)
                        self.assertEqual(output[1], {"metric": 1})
                    policy.train()
                    module.save_checkpoint(Path(root) / f"{step:06}", step, cfg, policy)
                root = Path(root)
                self.assertEqual((root / "best").resolve(), (root / "000006").resolve())
                self.assertEqual((root / "best/pretrained_model/model.safetensors").read_text(), "6")
                metadata = json.loads((root / "best.json").read_text())
                self.assertEqual(metadata["step"], 6)
                self.assertLess(metadata["eval_loss"], 2.0)
                records = [json.loads(line) for line in (root / "validation_losses.jsonl").read_text().splitlines()]
                self.assertEqual([row["step"] for row in records], [2, 4, 6])
                self.assertTrue((root / "000008/pretrained_model/model.safetensors").is_file())

    def test_non_validation_save_does_not_reuse_previous_loss(self):
        with tempfile.TemporaryDirectory() as root:
            def save_checkpoint(checkpoint_dir, step, cfg, policy):
                Path(checkpoint_dir).mkdir()

            module = SimpleNamespace(make_policy=lambda: LossPolicy(), save_checkpoint=save_checkpoint)
            install_best_checkpoint(module)
            policy = module.make_policy()
            cfg = SimpleNamespace(eval_steps=2)
            policy.eval()
            policy(1.0)
            policy.train()
            module.save_checkpoint(checkpoint_dir=Path(root) / "000002", step=2, cfg=cfg, policy=policy)
            module.save_checkpoint(checkpoint_dir=Path(root) / "000003", step=3, cfg=cfg, policy=policy)
            self.assertEqual((Path(root) / "best").resolve().name, "000002")
            self.assertEqual(len((Path(root) / "validation_losses.jsonl").read_text().splitlines()), 1)

    def test_failed_save_keeps_previous_best(self):
        with tempfile.TemporaryDirectory() as root:
            def save_checkpoint(checkpoint_dir, step, cfg, policy):
                raise OSError("disk full")

            module = SimpleNamespace(make_policy=lambda: LossPolicy(), save_checkpoint=save_checkpoint)
            install_best_checkpoint(module)
            policy = module.make_policy()
            policy.eval()
            policy(1.0)
            policy.train()
            with self.assertRaisesRegex(OSError, "disk full"):
                module.save_checkpoint(Path(root) / "000002", 2, SimpleNamespace(eval_steps=2), policy)
            self.assertFalse((Path(root) / "best").is_symlink())

    def test_save_and_eval_frequencies_must_align_unless_disabled(self):
        with tempfile.TemporaryDirectory() as root:
            common = ["--dataset-root", root, "--output-dir", str(Path(root) / "run"),
                      "--eval-steps", "1000", "--save-freq", "10000"]
            with self.assertRaisesRegex(ValueError, "each validation step"):
                build_cli(make_parser().parse_args(common), auxiliary=False)
            for extra in (["--no-save-best"], ["--save-freq", "500"], ["--smoke-test"]):
                build_cli(make_parser().parse_args(common + extra), auxiliary=False)

    def test_real_act_checkpoint_can_be_reloaded_from_best(self):
        from lerobot.common.train_utils import save_checkpoint
        from lerobot.configs.default import DatasetConfig
        from lerobot.configs.policies import PolicyFeature
        from training.ACT.act_training_config import ACTTrainPipelineConfig
        from lerobot.configs.types import FeatureType
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        config = ACTConfig(
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(9,)),
                "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(2,)),
            },
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
            device="cpu", pretrained_backbone_weights=None, use_vae=False,
            chunk_size=2, n_action_steps=2, dim_model=32, n_heads=4,
            dim_feedforward=64, n_encoder_layers=1, n_decoder_layers=1,
        )
        module = SimpleNamespace(make_policy=lambda: ACTPolicy(config), save_checkpoint=save_checkpoint)
        install_best_checkpoint(module)
        policy = module.make_policy()
        batch = {
            "observation.state": torch.zeros(1, 9),
            "observation.environment_state": torch.zeros(1, 2),
            "action": torch.zeros(1, 2, 7), "action_is_pad": torch.zeros(1, 2, dtype=torch.bool),
        }
        policy.eval()
        with torch.no_grad():
            loss, _ = policy.forward(batch)
        policy.train()
        with tempfile.TemporaryDirectory() as root:
            cfg = ACTTrainPipelineConfig(
                dataset=DatasetConfig(repo_id="local/test"), policy=config,
                output_dir=Path(root), steps=2, eval_steps=2,
            )
            module.save_checkpoint(
                checkpoint_dir=Path(root) / "000002", step=2, cfg=cfg, policy=policy,
                optimizer=torch.optim.AdamW(policy.parameters()),
            )
            best = Path(root) / "best/pretrained_model"
            self.assertNotIn("ema", json.loads((best / "train_config.json").read_text()))
            loaded = ACTPolicy.from_pretrained(str(best), local_files_only=True, strict=True)
            for key, value in policy.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key], value)
            self.assertAlmostEqual(json.loads((Path(root) / "best.json").read_text())["eval_loss"], loss.item())


if __name__ == "__main__":
    unittest.main()
