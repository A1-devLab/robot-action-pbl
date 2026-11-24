import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src import score_v2
from src.classify import EqualWeightClassifier
from src.cluster_v2 import AXES, NAMES, MechanismClusterer
from src.data import EXPECTED_COLUMNS, load_dataset, load_folder
from src.features_release import WINDOW, release_features
from src.phases import FINGERS, segment

ROOT = Path(__file__).resolve().parents[1]
GRIP = np.r_[np.zeros(20), np.linspace(0, 1.5708, 4)[1:], np.full(30, 1.5708),
             np.linspace(1.5708, 0, 4)[1:], np.zeros(20)]


def write_episode(path, number, grip, thumb):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        for frame, (g, t) in enumerate(zip(grip, thumb)):
            row = dict.fromkeys(EXPECTED_COLUMNS, 0.0)
            row.update(episode_index=number, frame_index=frame, timestamp=frame / 30,
                       **{f"state.{finger}": g for finger in FINGERS},
                       **{"state.l_hand_thumb_y": t, "state.l_sh_r": -frame / 100, "state.l_el_p": frame / 100})
            writer.writerow(row)


class ReleaseFeatureTests(unittest.TestCase):
    def test_thumb_opening_in_first_02s(self):
        with tempfile.TemporaryDirectory() as tmp:
            open_start = 53                                     # first frame after the closed run
            thumb = np.full(len(GRIP), 1.4)
            thumb[open_start:] = 1.4 - 0.1 * np.arange(len(GRIP) - open_start)
            write_episode(Path(tmp) / "a.csv", 1, GRIP, thumb)
            episode = load_folder(tmp)[0]
            self.assertEqual(segment(episode).open_start, open_start)
            self.assertAlmostEqual(release_features(episode)["thumb_open_02s"], 0.1 * WINDOW)


class ScoreV2Tests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.F = pd.DataFrame(rng.normal(size=(40, 4)), columns=list(score_v2.SCORE_FEATURES))
        self.model = EqualWeightClassifier(directions=dict(score_v2.DIRECTIONS), groups=dict(score_v2.GROUPS)).fit(self.F)

    def test_breakdown_and_release_direction(self):
        table = self.model.breakdown(self.F)
        self.assertEqual(list(table.columns), ["sub_carry", "sub_smooth", "sub_release", "score"])
        mean = pd.DataFrame([self.F.mean()])
        faster = mean.copy()
        faster["thumb_open_02s"] += 1.0                         # thumb snaps open -> worse
        self.assertAlmostEqual(self.model.score(mean).iloc[0], 50.0)
        self.assertLess(self.model.score(faster).iloc[0], 50.0)
        self.assertLess(self.model.breakdown(faster)["sub_release"].iloc[0], 50.0)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = EqualWeightClassifier.load(self.model.save(Path(tmp) / "p.json"))
        np.testing.assert_allclose(loaded.breakdown(self.F), self.model.breakdown(self.F))


class MechanismClustererTests(unittest.TestCase):
    def test_names_follow_cluster_centres(self):
        rng = np.random.default_rng(0)
        centres = {"clean": (1.0, 0.5, 1.0), "abrupt": (0.5, 0.5, -1.5), "hesitant": (-1.5, -1.0, 0.0)}
        Z = pd.DataFrame(np.vstack([np.array(c) + rng.normal(0, 0.15, (12, 3)) for c in centres.values()]),
                         columns=list(AXES))
        clusters = MechanismClusterer().fit(Z).predict(Z)
        self.assertEqual(clusters[:12].tolist(), [0] * 12)
        self.assertEqual(clusters[12:24].tolist(), [1] * 12)
        self.assertEqual(clusters[24:].tolist(), [2] * 12)
        self.assertEqual(len(set(NAMES.values())), 3)


@unittest.skipUnless((ROOT / "data/success").is_dir(), "real dataset not present")
class RealDataTests(unittest.TestCase):
    def test_v2_on_reference_data(self):
        episodes = load_dataset(ROOT / "data")
        F = pd.DataFrame([score_v2.score_inputs_v2(e) for e in episodes])
        y = np.array([e.label for e in episodes])
        model = score_v2.load()
        refit = score_v2.fit_reference(episodes)
        np.testing.assert_allclose(model.score(F), refit.score(F))   # params.json matches the data
        self.assertGreater(((model.score(F) >= 50).astype(int) == y).mean(), 0.75)
        clusters = MechanismClusterer().fit(model.group_z(F)).predict(model.group_z(F))
        self.assertEqual(np.bincount(clusters).tolist(), [19, 18, 11])
        self.assertGreater(y[clusters == 0].mean(), y[clusters == 2].mean())


if __name__ == "__main__":
    unittest.main()
