"""Reproducible episode-level, out-of-fold logistic regression baseline."""

import argparse
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_dataset, write_summary
from .features import extract_features


def evaluate(X, y, seed=42):
    if y.value_counts().min() < 5 or len(y.unique()) != 2:
        raise ValueError("Five-fold stratification requires at least five episodes per class")
    pipeline = make_pipeline(VarianceThreshold(), StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=5000, random_state=seed))
    predictions = np.empty(len(y), dtype=int)
    probabilities = np.empty(len(y))
    dummy_predictions = np.empty(len(y), dtype=int)
    folds = np.empty(len(y), dtype=int)
    fold_results = []
    for fold, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y), 1):
        model = clone(pipeline).fit(X.iloc[train], y.iloc[train])
        predictions[test] = model.predict(X.iloc[test])
        probabilities[test] = model.predict_proba(X.iloc[test])[:, 1]
        dummy = DummyClassifier(strategy="most_frequent").fit(X.iloc[train], y.iloc[train])
        dummy_predictions[test] = dummy.predict(X.iloc[test])
        folds[test] = fold
        fold_results.append({"fold": fold, "n_train": len(train), "n_test": len(test),
                             "accuracy": accuracy_score(y.iloc[test], predictions[test]),
                             "f1": f1_score(y.iloc[test], predictions[test], zero_division=0)})
    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "precision_success": precision_score(y, predictions, zero_division=0),
        "recall_success": recall_score(y, predictions, zero_division=0),
        "f1_success": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "dummy_accuracy": accuracy_score(y, dummy_predictions),
        "folds": fold_results,
    }
    return metrics, pd.DataFrame({"fold": folds, "label": y, "prediction": predictions,
                                  "probability_success": probabilities})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/baseline"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    episodes = load_dataset(args.data_dir)
    X, y, metadata = extract_features(episodes)
    metrics, predictions = evaluate(X, y, args.seed)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_summary(episodes, output / "episode_summary.csv")
    # Paths stay in the local inspection summary only, not in exported model tables.
    pd.concat([metadata[["episode_id"]], y, X], axis=1).to_csv(output / "features.csv", index=False)
    pd.concat([metadata[["episode_id"]], predictions], axis=1).to_csv(output / "oof_predictions.csv", index=False)
    metrics["config"] = {"seed": args.seed, "n_splits": 5, "n_episodes": len(y),
                         "n_features": X.shape[1], "C": 1.0, "threshold": 0.5,
                         "python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "sklearn": sklearn.__version__,
                         "evaluation": "out-of-fold; no hyperparameter search"}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(np.array(metrics["confusion_matrix"]), display_labels=["fail", "success"]).plot(ax=ax, colorbar=False)
    ax.set_title("Logistic regression: out-of-fold predictions")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
