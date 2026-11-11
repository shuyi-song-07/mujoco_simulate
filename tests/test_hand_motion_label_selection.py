"""Both auxiliary trainers explicitly select the matching value/mask pair."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from training.ACT import train_act_variation_baseline as act
from training.Diffusion import train_diffusion_variation_baseline as diffusion


class HandMotionLabelSelectionTest(unittest.TestCase):
    def test_window_selection_requires_new_fields_without_falling_back(self):
        for trainer in (act, diffusion):
            with self.subTest(trainer=trainer.__name__), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "meta").mkdir()
                (root / "data").mkdir()
                (root / "data/test.parquet").touch()
                metadata = {"total_frames": 6, "total_episodes": 2,
                            "features": {k: {} for k in (
                                "observation.state", "action", trainer.IMAGE_KEY,
                                trainer.GESTURE_KEY, trainer.HAND_DELTA_KEY, trainer.HAND_VALID_KEY,
                            )}}
                info_path = root / "meta/info.json"
                info_path.write_text(json.dumps(metadata))
                trainer.check_dataset(root, auxiliary=True, motion=True)
                with self.assertRaisesRegex(ValueError, "hand_delta_window"):
                    trainer.check_dataset(root, auxiliary=True, motion=True,
                                          hand_delta_source="window")
                metadata["features"].update({"observation.hand_delta_window_uv": {},
                                              "observation.hand_delta_window_valid": {}})
                info_path.write_text(json.dumps(metadata))
                args = trainer.make_parser(auxiliary=True, motion=True).parse_args([
                    "--dataset-root", str(root), "--hand-delta-source", "window",
                    "--hand-delta-weight", "0.1", "--gesture-weight", "0", "--print-command",
                ])
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    trainer.run(args, auxiliary=True)
                self.assertEqual(json.loads(output.getvalue())["hand_delta_source"], "window")
                self.assertEqual(json.loads(output.getvalue())["gesture_weight"], 0)


if __name__ == "__main__":
    unittest.main()
