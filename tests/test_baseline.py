import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import EXPECTED_COLUMNS, load_dataset
from src.features import extract_features
from src.train import evaluate


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name, number in (("fail", 1), ("success", 2)):
            (self.root / name).mkdir()
            self.write(name, number)

    def write(self, name, number, bad=False):
        with (self.root / name / f"episode_{number:06}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EXPECTED_COLUMNS)
            writer.writeheader()
            for frame in range(2):
                row = dict.fromkeys(EXPECTED_COLUMNS, float(frame))
                row.update(episode_index=number, frame_index=frame, timestamp=frame / 30)
                if bad:
                    row["state.l_sh_p"] = ""
                writer.writerow(row)

    def test_labels_features_and_metadata_separation(self):
        episodes = load_dataset(self.root)
        X, y, metadata = extract_features(episodes)
        self.assertEqual(y.tolist(), [0, 1])
        self.assertEqual(X.shape, (2, 208))
        self.assertEqual(X.loc[0, "state.l_sh_p__mean"], .5)
        self.assertEqual(X.loc[0, "state.l_sh_p__std"], .5)
        self.assertEqual(X.loc[0, "state.l_sh_p__range"], 1)
        self.assertEqual(X.loc[0, "state.l_sh_p__delta"], 1)
        self.assertTrue(all(c.startswith(("state.", "action.")) for c in X.columns))
        self.assertEqual(metadata.columns.tolist(), ["episode_id", "path"])

    def test_invalid_values_are_reported_and_rejected(self):
        self.write("fail", 1, bad=True)
        with self.assertRaises(ValueError):
            load_dataset(self.root)
        episode = load_dataset(self.root, strict=False)[0]
        self.assertEqual(episode.summary["missing_count"], 2)
        with self.assertRaises(ValueError):
            episode.motion_inputs()

    def test_duplicate_ids_across_classes_rejected(self):
        self.write("success", 1)
        with self.assertRaisesRegex(ValueError, "duplicate episode ID"):
            load_dataset(self.root)


class EvaluationTests(unittest.TestCase):
    def test_scaler_fits_training_rows_only_and_each_episode_is_held_out_once(self):
        X = pd.DataFrame({"signal": np.arange(20, dtype=float), "constant": np.ones(20)})
        y = pd.Series([0, 1] * 10)
        seen = []
        original = StandardScaler.fit

        def recording_fit(scaler, values, *args, **kwargs):
            seen.append(np.array(values))
            return original(scaler, values, *args, **kwargs)

        with patch.object(StandardScaler, "fit", recording_fit):
            metrics, predictions = evaluate(X, y)
        self.assertEqual(len(seen), 5)
        self.assertTrue(all(a.shape == (16, 1) for a in seen))
        for fold, training_values in enumerate(seen, 1):
            held_out = X.loc[predictions.fold == fold, "signal"].to_numpy()
            self.assertEqual(len(held_out), 4)
            self.assertEqual(len(np.intersect1d(training_values[:, 0], held_out)), 0)
        self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 20)
        self.assertTrue(np.isfinite(predictions.probability_success).all())


if __name__ == "__main__":
    unittest.main()
