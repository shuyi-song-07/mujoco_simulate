"""Keep held-out episodes out of Diffusion Policy normalization statistics."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from training.Diffusion.train_diffusion_variation_baseline import install_training_only_stats


class DiffusionTrainingStatsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data" / "chunk-000"
        self.data.mkdir(parents=True)
        self.cfg = SimpleNamespace(dataset=SimpleNamespace(root=self.root))

    def write_rows(self, name, episode_indices, states, actions):
        pq.write_table(
            pa.table({
                "episode_index": episode_indices,
                "observation.state": states,
                "action": actions,
            }),
            self.data / name,
        )

    @staticmethod
    def dataset(episodes, num_episodes):
        return SimpleNamespace(
            episodes=episodes,
            num_episodes=num_episodes,
            meta=SimpleNamespace(stats={"observation.images.overview": {"mean": [0.5]}}),
        )

    def fit(self, train, heldout):
        module = SimpleNamespace(make_train_eval_datasets=lambda cfg: (train, heldout))
        install_training_only_stats(module)
        return module.make_train_eval_datasets(self.cfg)

    def test_holdout_extremes_do_not_change_training_normalization(self):
        # Mix held-out and training rows in one file, and put another held-out
        # episode in its own file so neither file layout can leak evaluation.
        self.write_rows(
            "file-000.parquet", [10, 20, 10, 30],
            [[1.0, 2.0], [1000.0, -1000.0], [3.0, 6.0], [5.0, 4.0]],
            [[-2.0], [9999.0], [2.0], [6.0]],
        )
        self.write_rows("file-001.parquet", [20], [[-5000.0, 5000.0]], [[-9999.0]])
        train = self.dataset([10, 30], 2)
        heldout = self.dataset([20], 1)

        actual_train, actual_heldout = self.fit(train, heldout)

        self.assertIs(actual_train, train)
        self.assertIs(actual_heldout, heldout)
        expected_values = {
            "observation.state": np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 4.0]]),
            "action": np.array([[-2.0], [2.0], [6.0]]),
        }
        for key, values in expected_values.items():
            expected = {
                "mean": values.mean(axis=0), "std": values.std(axis=0),
                "min": values.min(axis=0), "max": values.max(axis=0),
                "count": [3],
            }
            for statistic, target in expected.items():
                np.testing.assert_allclose(train.meta.stats[key][statistic], target)
                np.testing.assert_allclose(heldout.meta.stats[key][statistic], target)
        self.assertEqual(train.meta.stats["observation.images.overview"], {"mean": [0.5]})
        # The two datasets use the same fitted values, but own their arrays.
        train.meta.stats["action"]["min"][0] = -123.0
        self.assertEqual(heldout.meta.stats["action"]["min"][0], -2.0)

    def test_no_holdout_fits_all_episodes(self):
        self.write_rows(
            "file-000.parquet", [0, 1],
            [[2.0, 4.0], [6.0, 8.0]], [[1.0], [3.0]],
        )
        train = self.dataset(None, 2)

        _, heldout = self.fit(train, None)

        self.assertIsNone(heldout)
        np.testing.assert_array_equal(train.meta.stats["observation.state"]["min"], [2.0, 4.0])
        np.testing.assert_array_equal(train.meta.stats["observation.state"]["max"], [6.0, 8.0])
        np.testing.assert_array_equal(train.meta.stats["action"]["count"], [2])

    def test_missing_training_rows_fail_instead_of_using_holdout(self):
        self.write_rows("file-000.parquet", [1], [[100.0, 200.0]], [[300.0]])

        with self.assertRaisesRegex(ValueError, "No training values found"):
            self.fit(self.dataset([0], 1), self.dataset([1], 1))


if __name__ == "__main__":
    unittest.main()
