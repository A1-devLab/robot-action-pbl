"""Demo day: classify (task 2), cluster (task 3) and score (task 4) every CSV in a folder.

    python -m src.demo --data-dir <folder> [--output reports/demo/results.csv]

Input   : a folder of CSVs with any file names (labels unknown), or a folder with
          success/ and fail/ (labels known -> accuracy is printed as well).
Task 2  : success if quality score >= 50 (src.classify, fixed threshold).
Task 3  : PhaseClusterer(k=3) is fitted on the reference episodes in data/ at start-up
          (deterministic, about a second) and applied to the new episodes. No model files.
Task 4  : fixed formula and reference statistics in reports/score/params.json.
Status  : ok | no_grasp (hand never closes -> fail, score 0) | invalid (file check failed)
          | error (unexpected problem in one file). One bad file never stops the run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from .classify import DEFAULT_THRESHOLD, predict_from_score
from .cluster import CLUSTER_INPUTS, PhaseClusterer, cluster_inputs
from .data import Episode, load_dataset, load_folder
from .features_phase import episode_features
from .phases import segment
from .score import DEFAULT_PARAMS, SCORE_FEATURES, QualityScore

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "data"
NO_GRASP_CLUSTER = -1
NO_GRASP_NAME = "no grasp"
SHOWN = ["episode_id", "status", "prediction", "score", "sub_carry", "sub_smooth", "sub_exit",
         "cluster", "cluster_name"]


def episode_inputs(episode: Episode) -> dict[str, float]:
    """10 phase features plus phase durations t_reach ... t_retract (raises if no grasp)."""
    values = episode_features(episode)
    values.update({f"t_{name}": seconds for name, seconds in segment(episode).seconds().items()})
    return {name: float(value) for name, value in values.items()}


def run(episodes: list[Episode], score_model: QualityScore, clusterer: PhaseClusterer | None,
        threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """One row per episode with status, prediction, score, sub-scores and cluster."""
    rows = []
    for episode in episodes:
        row = {"episode_id": episode.episode_id, "label": episode.label, "status": "ok"}
        if episode.summary["issues"]:
            row["status"] = f"invalid: {episode.summary['issues']}"
        else:
            try:
                row.update(episode_inputs(episode))
            except ValueError as error:
                row["status"] = "no_grasp" if "never closes" in str(error) else f"error: {error}"
            except Exception as error:  # the live demo must finish; report the file instead
                row["status"] = f"error: {type(error).__name__}: {error}"
        rows.append(row)

    table = pd.DataFrame(rows)
    for name in set(SCORE_FEATURES) | set(CLUSTER_INPUTS):
        if name not in table:
            table[name] = np.nan
    ok = table["status"] == "ok"
    no_grasp = table["status"] == "no_grasp"

    scores = pd.DataFrame(index=table.index, columns=["sub_carry", "sub_smooth", "sub_exit", "score"], dtype=float)
    if ok.any():
        scores.loc[ok] = score_model.breakdown(table.loc[ok, list(SCORE_FEATURES)]).astype(float)
    scores.loc[no_grasp, "score"] = 0.0
    table = table.join(scores)

    has_score = ok | no_grasp
    table["prediction"] = pd.Series(pd.NA, index=table.index, dtype="object")
    table.loc[has_score, "prediction"] = np.where(
        predict_from_score(table.loc[has_score, "score"], threshold) == 1, "success", "fail")

    table["cluster"] = pd.Series(pd.NA, index=table.index, dtype="Int64")
    table["cluster_name"] = pd.Series(pd.NA, index=table.index, dtype="object")
    if clusterer is not None and ok.any():
        clusters = clusterer.predict(table.loc[ok, list(CLUSTER_INPUTS)])
        table.loc[ok, "cluster"] = clusters
        table.loc[ok, "cluster_name"] = [clusterer.name(c) for c in clusters]
    table.loc[no_grasp, "cluster"] = NO_GRASP_CLUSTER
    table.loc[no_grasp, "cluster_name"] = NO_GRASP_NAME

    if (table["label"] >= 0).any():
        known = (table["label"] >= 0) & has_score
        table["correct"] = pd.Series(pd.NA, index=table.index, dtype="boolean")
        table.loc[known, "correct"] = (table.loc[known, "prediction"] == "success") == (table.loc[known, "label"] == 1)
    first = SHOWN[:3] + ["label"] + SHOWN[3:] + (["correct"] if "correct" in table else [])
    return table[first + [c for c in table.columns if c not in first]]


def fit_reference_clusterer(reference_dir: Path) -> PhaseClusterer:
    """Fit the task-3 clusterer on the labeled reference episodes (labels are not used)."""
    F, _, _ = cluster_inputs(load_dataset(reference_dir))
    return PhaseClusterer(k=3).fit(F)


def summarize(table: pd.DataFrame) -> str:
    lines = ["status: " + ", ".join(f"{k} {v}" for k, v in
                                    table["status"].str.split(":").str[0].value_counts().items())]
    known = table[(table["label"] >= 0) & table["prediction"].notna()]
    if len(known):
        y_true = (known["label"] == 1).astype(int)
        y_pred = (known["prediction"] == "success").astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        lines += [f"accuracy {(y_true == y_pred).mean():.3f} ({int((y_true == y_pred).sum())}/{len(known)}), "
                  f"balanced accuracy {balanced_accuracy_score(y_true, y_pred):.3f}",
                  f"confusion matrix [rows true fail/success, cols predicted fail/success]: {cm.tolist()}",
                  "mean score by label (1 = success): "
                  + ", ".join(f"{int(k)}: {v:.1f}" for k, v in known.groupby("label")["score"].mean().items())]
        if known["cluster"].notna().any():
            lines += ["cluster x label:",
                      pd.crosstab(known["cluster"], known["label"]).to_string()]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify, cluster and score every CSV in a folder.")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="folder of CSVs, or a folder with success/ and fail/")
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR,
                        help="labeled reference episodes used to fit the clusterer (default: repo data/)")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output", type=Path, default=Path("reports/demo/results.csv"))
    args = parser.parse_args()

    episodes = load_folder(args.data_dir)
    score_model = QualityScore.load(args.params)
    try:
        clusterer = fit_reference_clusterer(args.reference_dir)
    except (OSError, ValueError) as error:
        print(f"WARNING: clustering skipped, reference data could not be used ({error})")
        clusterer = None

    table = run(episodes, score_model, clusterer, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.output, index=False)

    pd.set_option("display.width", 200)
    shown = SHOWN + (["label", "correct"] if (table["label"] >= 0).any() else [])
    view = table[[c for c in shown if c in table]].copy()
    view["status"] = view["status"].str.slice(0, 40)
    print(view.round(1).to_string(index=False))
    print()
    print(summarize(table))
    print(f"\nResults: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
