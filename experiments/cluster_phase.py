"""Task 3: cluster episodes on phase features and print evidence for the cluster meanings.

Outputs (reports/cluster/, not required for the PR):
  k_selection.csv, profile.csv, assignments.csv, pca_clusters.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.cluster import PhaseClusterer, choose_k, cluster_inputs, profile
from src.data import load_dataset


ROOT = Path(__file__).resolve().parents[1]


def main():
    episodes = load_dataset(ROOT / "data")
    F, y, ids = cluster_inputs(episodes)
    numbers = ids["episode_id"].str.extract(r"(\d+)")[0].astype(int)

    selection = choose_k(F)
    model = PhaseClusterer(k=3).fit(F)
    clusters = model.predict(F)
    prof = profile(F, clusters)

    table = pd.DataFrame({"cluster": clusters, "label": y, "episode": numbers})
    summary = table.groupby("cluster").agg(n=("label", "size"), success=("label", "sum"),
                                           episodes=("episode", lambda s: sorted(s.tolist())))
    summary["success_rate"] = (summary["success"] / summary["n"]).round(2)
    summary["name"] = [model.name(c) for c in summary.index]

    out = ROOT / "reports/cluster"
    out.mkdir(parents=True, exist_ok=True)
    selection.to_csv(out / "k_selection.csv", index=False)
    prof.to_csv(out / "profile.csv")
    ids.assign(cluster=clusters, cluster_name=[model.name(c) for c in clusters],
               label=y).to_csv(out / "assignments.csv", index=False)

    P = model.embed(F)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for c in range(model.k):
        axes[0].scatter(*P[clusters == c, :2].T, label=f"C{c}: {model.name(c)}", s=30)
    axes[0].set(title="Clusters (PC1-PC2)", xlabel="PC1", ylabel="PC2")
    axes[0].legend(fontsize=7)
    sc = axes[1].scatter(*P[:, :2].T, c=numbers, cmap="viridis", s=30)
    axes[1].set(title="Episode number (not used as input)", xlabel="PC1")
    fig.colorbar(sc, ax=axes[1])
    fig.tight_layout()
    fig.savefig(out / "pca_clusters.png", dpi=150)

    pd.set_option("display.width", 200)
    print("k selection (higher silhouette / bootstrap ARI = better):")
    print(selection.round(3).to_string(index=False))
    print(f"\nPCA components kept: {model.pca_.n_components_} "
          f"({model.pca_.explained_variance_ratio_.sum():.0%} variance)")
    print("\nclusters:")
    print(summary.to_string())
    print(f"\nARI(cluster, success label) = {adjusted_rand_score(y, clusters):.2f}")
    print("\nprofile (z = standardized mean):")
    print(prof.round(2).to_string())


if __name__ == "__main__":
    main()
