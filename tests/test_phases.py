import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.data import EXPECTED_COLUMNS, load_dataset
from src.features_phase import FEATURE_NAMES, count_runs, extract_phase_features
from src.phases import FINGERS, count_grasps, segment

ROOT = Path(__file__).resolve().parents[1]


class SyntheticPhaseTests(unittest.TestCase):
    """Fingers open (0) -> ramp to pi/2 -> closed -> ramp back -> open."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.grip = np.r_[np.zeros(20), np.linspace(0, 1.5708, 4)[1:], np.full(30, 1.5708),
                          np.linspace(1.5708, 0, 4)[1:], np.zeros(20)]
        for name, number in (("fail", 1), ("success", 2)):
            (self.root / name).mkdir()
            self.write(name, number, self.grip)

    def write(self, name, number, grip):
        with (self.root / name / f"episode_{number:06}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EXPECTED_COLUMNS)
            writer.writeheader()
            for frame, g in enumerate(grip):
                row = dict.fromkeys(EXPECTED_COLUMNS, 0.0)
                row.update(episode_index=number, frame_index=frame, timestamp=frame / 30,
                           **{f"state.{finger}": g for finger in FINGERS},
                           **{"state.l_sh_r": frame / 100, "state.l_el_p": -frame / 100})
                writer.writerow(row)

    def test_segment_finds_hand_events(self):
        p = segment(load_dataset(self.root)[0])
        self.assertEqual((p.close_start, p.close_end, p.open_start, p.open_end), (19, 22, 53, 55))
        self.assertAlmostEqual(p.seconds()["hold"], 31 / 30)

    def test_features_shape_and_hold_delta(self):
        X, y, metadata = extract_phase_features(load_dataset(self.root))
        self.assertEqual(list(X.columns), list(FEATURE_NAMES))
        self.assertEqual(X.shape, (2, 10))
        self.assertAlmostEqual(X.loc[0, "delta_hold_l_sh_r"], (53 - 22) / 100)
        self.assertEqual(y.tolist(), [0, 1])
        self.assertNotIn("episode_id", X.columns)

    def test_hand_never_closes_is_rejected(self):
        self.write("fail", 1, np.zeros(60))
        with self.assertRaisesRegex(ValueError, "never closes"):
            segment(load_dataset(self.root)[0])

    def test_count_runs(self):
        self.assertEqual(count_runs(np.array([1, 1, 1, 0, 1, 1, 1, 1], bool), 3), 2)


@unittest.skipUnless((ROOT / "data/success").is_dir(), "real dataset not present")
class RealDataTests(unittest.TestCase):
    def test_every_episode_has_one_grasp(self):
        episodes = load_dataset(ROOT / "data")
        self.assertEqual(len(episodes), 48)
        self.assertEqual({count_grasps(e) for e in episodes}, {1})
        X, _, _ = extract_phase_features(episodes)
        self.assertFalse(X.isna().any().any())


if __name__ == "__main__":
    unittest.main()
