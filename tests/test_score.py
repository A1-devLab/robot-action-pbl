import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data import EXPECTED_COLUMNS, UNLABELED, load_folder
from src.phases import FINGERS
from src.score import SCORE_FEATURES, QualityScore, score_episodes

ROOT = Path(__file__).resolve().parents[1]


def write_episode(path, number, grip):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        for frame, g in enumerate(grip):
            row = dict.fromkeys(EXPECTED_COLUMNS, 0.0)
            row.update(episode_index=number, frame_index=frame, timestamp=frame / 30,
                       **{f"state.{finger}": g for finger in FINGERS},
                       **{"state.l_sh_r": -frame / 100, "state.l_el_p": frame / 100})
            writer.writerow(row)


GRIP = np.r_[np.zeros(20), np.linspace(0, 1.5708, 4)[1:], np.full(30, 1.5708),
             np.linspace(1.5708, 0, 4)[1:], np.zeros(20)]


class FormulaTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.F = pd.DataFrame(rng.normal(size=(40, 4)), columns=list(SCORE_FEATURES))
        self.model = QualityScore.fit(self.F)

    def test_reference_mean_scores_50(self):
        mean_row = pd.DataFrame([self.F.mean()])
        self.assertAlmostEqual(self.model.score(mean_row).iloc[0], 50.0)

    def test_directions_and_range(self):
        better = pd.DataFrame([self.F.mean()])
        better["delta_hold_l_el_p"] += 1.0     # +1: larger is better
        worse = pd.DataFrame([self.F.mean()])
        worse["rev_elp_hold"] += 1.0           # -1: larger is worse
        self.assertGreater(self.model.score(better).iloc[0], 50)
        self.assertLess(self.model.score(worse).iloc[0], 50)
        s = self.model.score(self.F * 100)
        self.assertTrue(((s >= 0) & (s <= 100)).all())

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = QualityScore.load(self.model.save(Path(tmp) / "p.json"))
        np.testing.assert_allclose(loaded.score(self.F), self.model.score(self.F))


class DemoFolderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_unlabeled_folder_free_names_no_grasp_and_invalid(self):
        write_episode(self.root / "new_a.csv", 7, GRIP)
        write_episode(self.root / "new_b.csv", 8, np.zeros(60))
        (self.root / "broken.csv").write_text("episode_index\n", encoding="utf-8")
        episodes = load_folder(self.root)
        self.assertEqual([e.label for e in episodes], [UNLABELED] * 3)
        table = score_episodes(episodes, QualityScore.load()).set_index("episode_id")
        self.assertEqual(table.loc["new_a", "status"], "ok")
        self.assertTrue(0 <= table.loc["new_a", "score"] <= 100)
        self.assertEqual(table.loc["new_b", "status"], "no_grasp")
        self.assertEqual(table.loc["new_b", "score"], 0.0)
        self.assertTrue(table.loc["broken", "status"].startswith("invalid"))
        self.assertTrue(np.isnan(table.loc["broken", "score"]))


@unittest.skipUnless((ROOT / "data/success").is_dir(), "dataset not present")
class RealDataTests(unittest.TestCase):
    def test_success_scores_higher_on_reference_data(self):
        table = score_episodes(load_folder(ROOT / "data"), QualityScore.load())
        self.assertTrue((table["status"] == "ok").all())
        means = table.groupby("label")["score"].mean()
        self.assertGreater(means[1], means[0] + 20)


if __name__ == "__main__":
    unittest.main()
