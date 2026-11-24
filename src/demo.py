"""Demo day: classify (task 2), cluster (task 3) and score (task 4) every CSV in a folder.

    python -m src.demo --data-dir <folder> [--version v2|v1] [--output reports/demo/results.csv]

Input   : a folder of CSVs with any file names (labels unknown), or a folder with
          success/ and fail/ (labels known -> accuracy is printed as well).
v2 (default)
  Task 4: quality score v2, carry + smooth + release (src.score_v2, reports/score_v2/params.json)
  Task 2: success if score >= 50 (5th model)
  Task 3: MechanismClusterer on the three v2 sub-scores (src.cluster_v2)
v1
  Task 4: PR #4 quality score (src.score, reports/score/params.json)
  Task 2: success if score >= 50 (4th model)
  Task 3: PR #3 PhaseClusterer (src.cluster)
Clusterers are fitted on the reference episodes in data/ at start-up (deterministic, about a
second); score statistics are fixed JSON files. No model files are needed.
Status  : ok | no_grasp (hand never closes -> fail, score 0) | invalid (file check failed)
          | error (unexpected problem in one file). One bad file never stops the run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from . import score_v2
from .classify import DEFAULT_THRESHOLD, EqualWeightClassifier, predict_from_score
from .cluster import CLUSTER_INPUTS, PhaseClusterer, cluster_inputs
from .cluster_v2 import MechanismClusterer
from .data import Episode, load_dataset, load_folder
from .features_phase import episode_features
from .features_release import release_features
from .phases import segment
from .score import DEFAULT_PARAMS as V1_PARAMS, SCORE_FEATURES as V1_FEATURES, QualityScore

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "data"
NO_GRASP_CLUSTER = -1
NO_GRASP_NAME = "no grasp"


@dataclass
class Pipeline:
    """What one model version needs from the shared per-episode inputs."""

    version: str
    inputs: tuple[str, ...]                                    # columns the pipeline reads
    breakdown: Callable[[pd.DataFrame], pd.DataFrame]          # -> sub_* columns + score
    sub_columns: tuple[str, ...]
    cluster: Callable[[pd.DataFrame], np.ndarray] | None = None
    cluster_name: Callable[[int], str] | None = None


def episode_inputs(episode: Episode) -> dict[str, float]:
    """10 phase features, phase durations t_reach ... t_retract and release features."""
    values = episode_features(episode)
    values.update({f"t_{name}": seconds for name, seconds in segment(episode).seconds().items()})
    values.update(release_features(episode))
    return {name: float(value) for name, value in values.items()}


def pipeline_v1(score_model: QualityScore, clusterer: PhaseClusterer | None) -> Pipeline:
    return Pipeline(
        "v1", tuple(dict.fromkeys(V1_FEATURES + CLUSTER_INPUTS)),
        lambda F: score_model.breakdown(F[list(V1_FEATURES)]), ("sub_carry", "sub_smooth", "sub_exit"),
        None if clusterer is None else (lambda F: clusterer.predict(F[list(CLUSTER_INPUTS)])),
        None if clusterer is None else clusterer.name)


def pipeline_v2(score_model: EqualWeightClassifier, clusterer: MechanismClusterer | None) -> Pipeline:
    features = list(score_v2.SCORE_FEATURES)
    return Pipeline(
        "v2", tuple(features), lambda F: score_model.breakdown(F[features]),
        tuple(f"sub_{g}" for g in score_v2.GROUPS),
        None if clusterer is None else (lambda F: clusterer.predict(score_model.group_z(F[features]))),
        None if clusterer is None else clusterer.name)


def build_pipeline(version: str = "v2", reference_dir: Path = REFERENCE_DIR,
                   params: Path | None = None) -> Pipeline:
    """Load the fixed score statistics and fit the clusterer on the reference episodes.
    If the reference episodes cannot be used, clustering is skipped with a warning."""
    if version == "v1":
        score_model = QualityScore.load(params or V1_PARAMS)
    elif version == "v2":
        score_model = score_v2.load(params or score_v2.DEFAULT_PARAMS)
    else:
        raise ValueError(f"Unknown version {version!r}")
    try:
        reference = load_dataset(reference_dir)
        if version == "v1":
            F, _, _ = cluster_inputs(reference)
            clusterer = PhaseClusterer(k=3).fit(F)
        else:
            F = pd.DataFrame([score_v2.score_inputs_v2(e) for e in reference])
            clusterer = MechanismClusterer().fit(score_model.group_z(F))
    except (OSError, ValueError) as error:
        print(f"WARNING: clustering skipped, reference data could not be used ({error})")
        clusterer = None
    return pipeline_v1(score_model, clusterer) if version == "v1" else pipeline_v2(score_model, clusterer)


def run(episodes: list[Episode], pipeline: Pipeline, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
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
    for name in pipeline.inputs:
        if name not in table:
            table[name] = np.nan
    ok = table["status"] == "ok"
    no_grasp = table["status"] == "no_grasp"

    scores = pd.DataFrame(index=table.index, columns=list(pipeline.sub_columns) + ["score"], dtype=float)
    if ok.any():
        scores.loc[ok] = pipeline.breakdown(table.loc[ok]).astype(float)
    scores.loc[no_grasp, "score"] = 0.0
    table = table.join(scores)

    has_score = ok | no_grasp
    table["prediction"] = pd.Series(pd.NA, index=table.index, dtype="object")
    table.loc[has_score, "prediction"] = np.where(
        predict_from_score(table.loc[has_score, "score"], threshold) == 1, "success", "fail")

    table["cluster"] = pd.Series(pd.NA, index=table.index, dtype="Int64")
    table["cluster_name"] = pd.Series(pd.NA, index=table.index, dtype="object")
    if pipeline.cluster is not None and ok.any():
        clusters = pipeline.cluster(table.loc[ok])
        table.loc[ok, "cluster"] = clusters
        table.loc[ok, "cluster_name"] = [pipeline.cluster_name(c) for c in clusters]
    table.loc[no_grasp, "cluster"] = NO_GRASP_CLUSTER
    table.loc[no_grasp, "cluster_name"] = NO_GRASP_NAME

    if (table["label"] >= 0).any():
        known = (table["label"] >= 0) & has_score
        table["correct"] = pd.Series(pd.NA, index=table.index, dtype="boolean")
        table.loc[known, "correct"] = (table.loc[known, "prediction"] == "success") == (table.loc[known, "label"] == 1)
    first = (["episode_id", "status", "prediction", "label", "score"] + list(pipeline.sub_columns)
             + ["cluster", "cluster_name"] + (["correct"] if "correct" in table else []))
    return table[first + [c for c in table.columns if c not in first]]


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
    parser.add_argument("--version", choices=["v2", "v1"], default="v2")
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR,
                        help="labeled reference episodes used to fit the clusterer (default: repo data/)")
    parser.add_argument("--params", type=Path, default=None, help="score statistics JSON (default per version)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output", type=Path, default=Path("reports/demo/results.csv"))
    args = parser.parse_args()

    episodes = load_folder(args.data_dir)
    pipeline = build_pipeline(args.version, args.reference_dir, args.params)
    table = run(episodes, pipeline, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.output, index=False)

    pd.set_option("display.width", 220)
    shown = (["episode_id", "status", "prediction", "score"] + list(pipeline.sub_columns) + ["cluster", "cluster_name"]
             + (["label", "correct"] if (table["label"] >= 0).any() else []))
    view = table[[c for c in shown if c in table]].copy()
    view["status"] = view["status"].str.slice(0, 40)
    print(f"model version: {pipeline.version}")
    print(view.round(1).to_string(index=False))
    print()
    print(summarize(table))
    print(f"\nResults: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
