import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src import score_v2
from src.cluster import CLUSTER_INPUTS, PhaseClusterer
from src.cluster_v2 import AXES, MechanismClusterer
from src.data import EXPECTED_COLUMNS, load_folder
from src.demo import NO_GRASP_CLUSTER, build_pipeline, pipeline_v1, pipeline_v2, run, summarize
from src.phases import FINGERS
from src.score import QualityScore

ROOT = Path(__file__).resolve().parents[1]
GRIP = np.r_[np.zeros(20), np.linspace(0, 1.5708, 4)[1:], np.full(30, 1.5708),
             np.linspace(1.5708, 0, 4)[1:], np.zeros(20)]


def write_episode(path, number, grip):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        for frame, g in enumerate(grip):
            row = dict.fromkeys(EXPECTED_COLUMNS, 0.0)
            row.update(episode_index=number, frame_index=frame, timestamp=frame / 30,
                       **{f"state.{finger}": g for finger in FINGERS},
                       **{"state.l_sh_r": -frame / 100, "state.l_el_p": frame / 100,
                          "state.l_hand_thumb_y": 1.4 - 0.02 * max(0, frame - 53)})
            writer.writerow(row)


def synthetic_v1_clusterer():
    rng = np.random.default_rng(0)
    data = np.vstack([c + rng.normal(0, 0.3, (15, 13)) for c in (3.0, 0.0, -3.0)])
    return PhaseClusterer(k=3).fit(pd.DataFrame(data, columns=list(CLUSTER_INPUTS)))


def synthetic_v2_clusterer():
    rng = np.random.default_rng(0)
    centres = [(1.0, 0.5, 1.0), (0.5, 0.5, -1.5), (-1.5, -1.0, 0.0)]
    Z = pd.DataFrame(np.vstack([np.array(c) + rng.normal(0, 0.15, (12, 3)) for c in centres]), columns=list(AXES))
    return MechanismClusterer().fit(Z)


class SyntheticDemoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        write_episode(self.root / "trial_a.csv", 7, GRIP)
        write_episode(self.root / "trial_b.csv", 8, np.zeros(60))
        (self.root / "broken.csv").write_text("episode_index\n", encoding="utf-8")

    def check(self, table, sub_columns):
        table = table.set_index("episode_id")
        ok = table.loc["trial_a"]
        self.assertEqual(ok["status"], "ok")
        self.assertEqual(ok["prediction"] == "success", ok["score"] >= 50)
        self.assertIn(int(ok["cluster"]), {0, 1, 2})
        for column in sub_columns:
            self.assertTrue(0 <= ok[column] <= 100)
        no_grasp = table.loc["trial_b"]
        self.assertEqual((no_grasp["status"], no_grasp["prediction"], no_grasp["score"]), ("no_grasp", "fail", 0.0))
        self.assertEqual(int(no_grasp["cluster"]), NO_GRASP_CLUSTER)
        broken = table.loc["broken"]
        self.assertTrue(broken["status"].startswith("invalid"))
        self.assertTrue(pd.isna(broken["prediction"]) and np.isnan(broken["score"]))
        self.assertNotIn("correct", table.columns)
        self.assertIn("status: ", summarize(table.reset_index()))

    def test_v2_pipeline(self):
        pipeline = pipeline_v2(score_v2.load(), synthetic_v2_clusterer())
        self.check(run(load_folder(self.root), pipeline), ("sub_carry", "sub_smooth", "sub_release"))

    def test_v1_pipeline(self):
        pipeline = pipeline_v1(QualityScore.load(), synthetic_v1_clusterer())
        self.check(run(load_folder(self.root), pipeline), ("sub_carry", "sub_smooth", "sub_exit"))

    def test_runs_without_clusterer(self):
        table = run(load_folder(self.root), pipeline_v2(score_v2.load(), None))
        self.assertTrue(table["cluster"].isna().sum() >= 2)          # ok and broken rows have no cluster
        self.assertEqual(table.set_index("episode_id").loc["trial_a", "status"], "ok")


@unittest.skipUnless((ROOT / "data/success").is_dir(), "real dataset not present")
class RealDataDemoTests(unittest.TestCase):
    def test_reference_data_end_to_end(self):
        episodes = load_folder(ROOT / "data")
        for version, min_accuracy, sizes in (("v2", 0.75, [11, 18, 19]), ("v1", 0.6, [10, 10, 28])):
            table = run(episodes, build_pipeline(version))
            self.assertTrue((table["status"] == "ok").all())
            self.assertGreater(table["correct"].astype(bool).mean(), min_accuracy)
            self.assertEqual(sorted(table["cluster"].astype(int).value_counts().tolist()), sizes)
            self.assertIn("accuracy", summarize(table))


if __name__ == "__main__":
    unittest.main()
