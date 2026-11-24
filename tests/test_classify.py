from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.classify import DEFAULT_THRESHOLD, EqualWeightClassifier, predict_from_score, select_score_inputs
from src.data import load_dataset
from src.score import DIRECTIONS, SCORE_FEATURES, QualityScore, score_inputs

ROOT = Path(__file__).resolve().parents[1]


def separable(seed=0):
    """a: strong +, b: near copy of a, c: strong -, e: weaker +, d: noise."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(30), np.ones(30)].astype(int)
    a = 2.0 * y + rng.normal(0, 0.6, 60)
    F = pd.DataFrame({"a": a, "b": a + rng.normal(0, 0.05, 60), "c": -1.5 * y + rng.normal(0, 0.6, 60),
                      "d": rng.normal(0, 1, 60), "e": 0.8 * y + rng.normal(0, 0.6, 60)})
    return F, y


class SelectionTests(unittest.TestCase):
    def test_picks_signal_skips_redundant_and_noise(self):
        F, y = separable()
        chosen = select_score_inputs(F, y, k=3, max_corr=0.9)
        self.assertEqual(len({"a", "b"} & set(chosen)), 1)      # only one of the near-duplicates
        self.assertEqual(chosen["c"], -1.0)                      # larger c goes with fail
        self.assertEqual(chosen["e"], 1.0)
        self.assertNotIn("d", chosen)

    def test_needs_both_classes(self):
        F, _ = separable()
        with self.assertRaises(ValueError):
            select_score_inputs(F, np.ones(len(F)))


class EqualWeightTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.F = pd.DataFrame(rng.normal(size=(40, 4)), columns=list(SCORE_FEATURES))

    def test_reproduces_quality_score(self):
        ours = EqualWeightClassifier(directions=dict(DIRECTIONS)).fit(self.F).score(self.F)
        np.testing.assert_allclose(ours, QualityScore.fit(self.F).score(self.F), atol=1e-9)

    def test_threshold_50_is_reference_average(self):
        model = EqualWeightClassifier(directions=dict(DIRECTIONS)).fit(self.F)
        mean = pd.DataFrame([self.F.mean()])
        self.assertAlmostEqual(model.score(mean).iloc[0], DEFAULT_THRESHOLD)
        better, worse = mean.copy(), mean.copy()
        better["delta_hold_l_el_p"] += 1.0   # direction +1
        worse["rev_elp_hold"] += 1.0         # direction -1
        self.assertEqual(model.predict(better)[0], 1)
        self.assertEqual(model.predict(worse)[0], 0)

    def test_chooses_inputs_when_directions_missing(self):
        F, y = separable()
        model = EqualWeightClassifier(k=2).fit(F, y)
        self.assertEqual(len(model.directions_), 2)
        self.assertGreater((model.predict(F) == y).mean(), 0.9)
        with self.assertRaises(ValueError):
            EqualWeightClassifier().fit(F)

    def test_predict_from_score(self):
        self.assertEqual(predict_from_score([49.9, 50.0, 80.0, np.nan]).tolist(), [0, 1, 1, 0])


@unittest.skipUnless((ROOT / "data/success").is_dir(), "real dataset not present")
class RealDataTests(unittest.TestCase):
    def test_score_rule_beats_majority_on_reference_data(self):
        episodes = load_dataset(ROOT / "data")
        F = pd.DataFrame([score_inputs(e) for e in episodes])
        y = np.array([e.label for e in episodes])
        prediction = predict_from_score(QualityScore.load().score(F))
        self.assertGreater((prediction == y).mean(), max(y.mean(), 1 - y.mean()))


if __name__ == "__main__":
    unittest.main()
