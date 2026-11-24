"""Task 3 (v2): cluster episodes by how they carry and release the bottle.

Input  : the three v2 sub-scores as mean oriented z (src.score_v2): carry, smooth, release
         (0 = reference average, larger = better). Labels are never used.
Model  : KMeans(k=3, n_init=50, random_state=0) on the three values.
Names  : assigned from the cluster centres, not from labels, so new data gets the same names
           - lowest carry                      -> 2 "hesitant carry, unstable placement"
           - of the rest, lowest release       -> 1 "good carry, abrupt hand opening"
           - remaining                         -> 0 "clean carry and gentle release"
Evidence for the names (videos of the 1st dataset, reports/video_evidence/): cluster 2 holds
most fails where the bottle was released high or tilted (type A); cluster 1 holds fails of
both types with a fast-opening hand; cluster 0 has the highest success rate.
Compared with src.cluster (PR #3), these clusters depend much less on recording order
(reports/fifth_model/README.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

AXES = ("carry", "smooth", "release")
NAMES = {
    0: "clean carry and gentle release",
    1: "good carry, abrupt hand opening",
    2: "hesitant carry, unstable placement",
}


@dataclass
class MechanismClusterer:
    seed: int = 0

    def fit(self, Z: pd.DataFrame) -> "MechanismClusterer":
        X = Z[list(AXES)].to_numpy(dtype=float)
        self.kmeans_ = KMeans(3, n_init=50, random_state=self.seed).fit(X)
        centres = pd.DataFrame(self.kmeans_.cluster_centers_, columns=list(AXES))
        hesitant = int(centres["carry"].idxmin())
        rest = centres.drop(index=hesitant)
        abrupt = int(rest["release"].idxmin())
        clean = int(rest.drop(index=abrupt).index[0])
        self.relabel_ = {clean: 0, abrupt: 1, hesitant: 2}
        self.centres_ = centres.rename(index=self.relabel_).sort_index()
        return self

    def predict(self, Z: pd.DataFrame) -> np.ndarray:
        raw = self.kmeans_.predict(Z[list(AXES)].to_numpy(dtype=float))
        return np.array([self.relabel_[int(c)] for c in raw])

    @staticmethod
    def name(cluster: int) -> str:
        return NAMES[int(cluster)]
