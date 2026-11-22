"""Label-free clustering of episodes on phase features (task 3).

Input  : 10 phase features + reach/hold/retract durations (13 values per episode)
Model  : StandardScaler -> PCA (80% variance) -> KMeans(k)
Labels : success/fail are never used for fitting; they are only cross-tabulated afterwards.
Cluster numbers are reordered by the mean shoulder roll at grasp (high -> low) so the
same group keeps the same number and name across runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .data import Episode
from .features_phase import FEATURE_NAMES, extract_phase_features

TIMING = ("t_reach", "t_hold", "t_retract")
CLUSTER_INPUTS = FEATURE_NAMES + TIMING
ORDER_BY = "pose_grasp_l_sh_r"
NAMES_K3 = {
    0: "skilled: wide grasp, long carry, quick exit",
    1: "short carry, release close with bent elbow",
    2: "early hesitant: low grasp, many pauses",
}


def cluster_inputs(episodes: list[Episode]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """13 cluster inputs, labels (for reporting only) and episode ids."""
    X, y, metadata = extract_phase_features(episodes)
    F = pd.concat([X, metadata[list(TIMING)]], axis=1)[list(CLUSTER_INPUTS)]
    return F, y, metadata[["episode_id"]]


@dataclass
class PhaseClusterer:
    k: int = 3
    variance: float = 0.80
    seed: int = 0
    names: dict[int, str] = field(default_factory=dict)

    def fit(self, F: pd.DataFrame) -> "PhaseClusterer":
        self.scaler_ = StandardScaler().fit(F)
        Z = self.scaler_.transform(F)
        full = PCA().fit(Z)
        n = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), self.variance) + 1)
        self.pca_ = PCA(n, random_state=self.seed).fit(Z)
        self.kmeans_ = KMeans(self.k, n_init=50, random_state=self.seed).fit(self.pca_.transform(Z))
        raw = self.kmeans_.labels_
        order = F.groupby(raw)[ORDER_BY].mean().sort_values(ascending=False).index
        self.relabel_ = {old: new for new, old in enumerate(order)}
        if not self.names and self.k == 3:
            self.names = dict(NAMES_K3)
        return self

    def embed(self, F: pd.DataFrame) -> np.ndarray:
        return self.pca_.transform(self.scaler_.transform(F[list(CLUSTER_INPUTS)]))

    def predict(self, F: pd.DataFrame) -> np.ndarray:
        raw = self.kmeans_.predict(self.embed(F))
        return np.array([self.relabel_[c] for c in raw])

    def name(self, cluster: int) -> str:
        return self.names.get(int(cluster), f"cluster {cluster}")


def profile(F: pd.DataFrame, clusters: np.ndarray) -> pd.DataFrame:
    """Per-cluster mean of each input, raw and standardized (z), for interpretation."""
    z = (F - F.mean()) / F.std(ddof=0)
    raw = F.groupby(clusters).mean().T.add_prefix("mean_c")
    zs = z.groupby(clusters).mean().T.add_prefix("z_c")
    return pd.concat([zs, raw], axis=1)


def choose_k(F: pd.DataFrame, ks=range(2, 7), seed: int = 0, n_boot: int = 50) -> pd.DataFrame:
    """Silhouette and bootstrap stability (ARI vs. full-data clustering) for each k."""
    rows = []
    rng = np.random.default_rng(seed)
    for k in ks:
        model = PhaseClusterer(k=k, seed=seed).fit(F)
        P, base = model.embed(F), model.predict(F)
        boots = []
        for b in range(n_boot):
            idx = rng.choice(len(F), len(F), replace=True)
            km = KMeans(k, n_init=10, random_state=b).fit(P[idx])
            boots.append(adjusted_rand_score(base, km.predict(P)))
        rows.append({"k": k, "silhouette": silhouette_score(P, base),
                     "bootstrap_ari": float(np.mean(boots)),
                     "smallest_cluster": int(np.bincount(base).min())})
    return pd.DataFrame(rows)
