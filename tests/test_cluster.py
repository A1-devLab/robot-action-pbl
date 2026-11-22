from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.cluster import CLUSTER_INPUTS, ORDER_BY, PhaseClusterer, choose_k, cluster_inputs, profile
from src.data import load_dataset

ROOT = Path(__file__).resolve().parents[1]


def blobs(seed=0):
    """Three well-separated synthetic groups in the 13 cluster inputs."""
    rng = np.random.default_rng(seed)
    centers = np.array([[3.0] * 13, [0.0] * 13, [-3.0] * 13])
    data = np.vstack([c + rng.normal(0, 0.3, (15, 13)) for c in centers])
    return pd.DataFrame(data, columns=list(CLUSTER_INPUTS))


class SyntheticClusterTests(unittest.TestCase):
    def test_recovers_groups_and_orders_them(self):
        F = blobs()
        clusters = PhaseClusterer(k=3).fit(F).predict(F)
        self.assertEqual(clusters[:15].tolist(), [0] * 15)    # highest ORDER_BY -> cluster 0
        self.assertEqual(clusters[15:30].tolist(), [1] * 15)
        self.assertEqual(clusters[30:].tolist(), [2] * 15)

    def test_predict_is_deterministic_and_named(self):
        F = blobs()
        a = PhaseClusterer(k=3).fit(F).predict(F)
        model = PhaseClusterer(k=3).fit(F)
        self.assertTrue(np.array_equal(a, model.predict(F)))
        self.assertEqual(len({model.name(c) for c in range(3)}), 3)

    def test_profile_and_choose_k(self):
        F = blobs()
        clusters = PhaseClusterer(k=3).fit(F).predict(F)
        prof = profile(F, clusters)
        self.assertGreater(prof.loc[ORDER_BY, "z_c0"], prof.loc[ORDER_BY, "z_c2"])
        table = choose_k(F, ks=range(2, 5), n_boot=5)
        self.assertEqual(table["k"].tolist(), [2, 3, 4])
        self.assertEqual(table.loc[table["silhouette"].idxmax(), "k"], 3)


@unittest.skipUnless((ROOT / "data/success").is_dir(), "real dataset not present")
class RealDataClusterTests(unittest.TestCase):
    def test_three_clusters_without_labels(self):
        F, y, ids = cluster_inputs(load_dataset(ROOT / "data"))
        self.assertEqual(list(F.columns), list(CLUSTER_INPUTS))
        self.assertNotIn("label", F.columns)
        clusters = PhaseClusterer(k=3).fit(F).predict(F)
        self.assertEqual(sorted(np.bincount(clusters).tolist()), [10, 10, 28])


if __name__ == "__main__":
    unittest.main()
