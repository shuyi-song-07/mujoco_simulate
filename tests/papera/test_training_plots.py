"""Training analysis supports official ACT with gradient clipping disabled."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from training.papera import plot_training_analysis as plots
except ModuleNotFoundError as error:
    if error.name != "matplotlib":
        raise
    plots = None


@unittest.skipIf(plots is None, "Matplotlib is available in mujoco_repro")
class TrainingPlotTests(unittest.TestCase):
    def render(self, clip_limit):
        captured = {}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            config = {"model": {"chunk_size": 30, "latent_dim": 32},
                      "gradient_clip_norm": clip_limit, "log_every": 25}
            (run / "run.json").write_text(json.dumps({"provenance": {
                "config": config, "variant": "baseline",
                "model_implementation": "official-act-detrvae-papera-v1"}}), encoding="utf-8")
            rows = [{"event": "train", "step": 1, "action": .4, "total": 2.4,
                     "kl": .2, "gradient_norm": 5., "lr": 1e-5},
                    {"event": "validation", "step": 1, "prior_action_l1": .5,
                     "xyz_error_mm": 10., "gripper_error_mm": 3., "rpy_mae_degrees": 2.}]
            (run / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")

            def capture(fig, output, stem, description, manifest):
                captured[stem] = {"titles": [axis.get_title() for axis in fig.axes],
                                  "labels": [[line.get_label() for line in axis.lines]
                                             for axis in fig.axes],
                                  "description": description}
                plots.plt.close(fig)

            with patch.object(plots, "save_figure", side_effect=capture):
                plots.log_figures(run, run, {"sources": [], "artifacts": []})
        return captured

    def test_disabled_clip_renders_without_a_clip_limit(self):
        rendered = self.render(None)
        diagnostic = rendered["02_optimization_diagnostics"]
        self.assertIn("clipping disabled", diagnostic["titles"][2])
        self.assertEqual(len(diagnostic["labels"][2]), 1)
        self.assertIn("Gradient clipping is disabled", diagnostic["description"])
        labels = rendered["01_learning_curves"]["labels"][0]
        self.assertIn("Train: posterior, full B*K*A mean", labels)
        self.assertIn("Validation: prior z=0, valid coordinates", labels)

    def test_historical_clip_limit_remains_visible(self):
        diagnostic = self.render(1.)["02_optimization_diagnostics"]
        self.assertIn("Clip limit = 1", diagnostic["labels"][2])
        self.assertNotIn("clipping disabled", diagnostic["titles"][2])


if __name__ == "__main__":
    unittest.main()
