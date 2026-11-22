"""Check the task-4 score: does it rank success above fail on unseen episodes, and is it
more than recording order?

Protocol: StratifiedKFold(5, shuffle) x seeds 0-9. Reference statistics (mean/std) and,
for the logistic comparison, the weights are fitted on the training folds only.
Note: the four inputs were chosen after looking at single-feature AUCs on all 48
episodes, so these CV numbers are slightly optimistic.

Outputs (reports/score/): validation.json, score_by_label.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.cluster import PhaseClusterer, cluster_inputs
from src.data import load_dataset
from src.score import SCORE_FEATURES, QualityScore, score_inputs


ROOT = Path(__file__).resolve().parents[1]


def cv_scores(F, y, method, seed):
    oof = np.zeros(len(y))
    for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(F, y):
        if method == "equal_weights":
            oof[test] = QualityScore.fit(F.iloc[train]).raw(F.iloc[test])
        else:
            scaler = StandardScaler().fit(F.iloc[train])
            model = LogisticRegression(C=1.0).fit(scaler.transform(F.iloc[train]), y.iloc[train])
            oof[test] = model.decision_function(scaler.transform(F.iloc[test]))
    return oof


def main():
    episodes = load_dataset(ROOT / "data")
    F = pd.DataFrame([score_inputs(e) for e in episodes])
    y = pd.Series([e.label for e in episodes], name="label")
    number = pd.Series([int(e.episode_id[-6:]) for e in episodes])
    early = (number <= 24).to_numpy()

    result = {"single_feature_auc": {c: round(roc_auc_score(y, F[c]), 3) for c in SCORE_FEATURES},
              "episode_number_auc": round(roc_auc_score(y, number), 3)}
    for method in ("equal_weights", "logistic"):
        runs = [cv_scores(F, y, method, s) for s in range(10)]
        auc = [roc_auc_score(y, r) for r in runs]
        mean_oof = np.mean(runs, axis=0)
        result[method] = {
            "cv_auc_mean": round(float(np.mean(auc)), 3), "cv_auc_std": round(float(np.std(auc)), 3),
            "auc_episodes_1_24": round(roc_auc_score(y[early], mean_oof[early]), 3),
            "auc_episodes_25_48": round(roc_auc_score(y[~early], mean_oof[~early]), 3),
            "spearman_with_episode_number": round(float(spearmanr(mean_oof, number)[0]), 3)}

    model = QualityScore.load()
    score = model.score(F)
    Fc, _, _ = cluster_inputs(episodes)
    clusters = PhaseClusterer(k=3).fit(Fc).predict(Fc)
    by_cluster = pd.DataFrame({"cluster": clusters, "label": y, "score": score})
    result["score_by_label"] = score.groupby(y).mean().round(1).rename({0: "fail", 1: "success"}).to_dict()
    result["score_by_cluster_and_label"] = {
        f"C{c}_{'success' if l else 'fail'}": round(v, 1)
        for (c, l), v in by_cluster.groupby(["cluster", "label"])["score"].mean().items()}

    out = ROOT / "reports/score"
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(0, 100, 11)
    ax.hist(score[y == 0], bins, alpha=0.6, label="fail")
    ax.hist(score[y == 1], bins, alpha=0.6, label="success")
    ax.set(title="Quality score of the 48 reference episodes", xlabel="score (0-100)", ylabel="episodes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "score_by_label.png", dpi=150)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
