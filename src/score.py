"""Task 4: quality score (0-100) of one pick-and-place episode.

Formula
    z_i   = d_i * (x_i - mean_i) / std_i, clipped to [-3, 3]
    s     = (z_1 + z_2 + z_3 + z_4) / 4                 (equal weights)
    score = 100 * Phi(s / std_s)                       (Phi = standard normal CDF)

    i  feature             d   meaning (from the phase features)
    1  delta_hold_l_sh_r   -1  shoulder roll swings inward while carrying (good carry)
    2  delta_hold_l_el_p   +1  elbow extends toward the box while carrying (good carry)
    3  rev_elp_hold        -1  elbow direction reversals while carrying (hesitation)
    4  t_retract           -1  seconds from release to the end (slow exit)

mean_i, std_i and std_s come from the 48 reference episodes and are stored in
reports/score/params.json, so the same formula scores any new CSV. 50 means the
reference average, 84 is one standard deviation better. Episodes whose hand never
closes get 0 (no grasp); invalid files get no score.

Why these four and equal weights: they separate success/fail most (AUC 0.74-0.81
each), are not redundant (pose_grasp_l_sh_r was dropped, r = -0.90 with feature 1),
and equal weights beat fitted logistic weights in repeated CV with only 48 episodes
(experiments/validate_score.py). Success/fail labels choose the directions only.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Episode, load_folder
from .features_phase import episode_features
from .phases import segment

DIRECTIONS = {
    "delta_hold_l_sh_r": -1.0,
    "delta_hold_l_el_p": 1.0,
    "rev_elp_hold": -1.0,
    "t_retract": -1.0,
}
SCORE_FEATURES = tuple(DIRECTIONS)
GROUPS = {"carry": ("delta_hold_l_sh_r", "delta_hold_l_el_p"),
          "smooth": ("rev_elp_hold",), "exit": ("t_retract",)}
CLIP = 3.0
DEFAULT_PARAMS = Path(__file__).resolve().parents[1] / "reports/score/params.json"


def score_inputs(episode: Episode) -> dict[str, float]:
    """The four raw score inputs of one valid episode (raises if no grasp)."""
    features = episode_features(episode)
    features["t_retract"] = segment(episode).seconds()["retract"]
    return {name: float(features[name]) for name in SCORE_FEATURES}


def _phi(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, float) / math.sqrt(2.0)))


@dataclass
class QualityScore:
    mean: dict[str, float]
    std: dict[str, float]
    std_s: float
    directions: dict[str, float]

    @classmethod
    def fit(cls, F: pd.DataFrame) -> "QualityScore":
        """Reference statistics from the raw inputs of the reference episodes."""
        F = F[list(SCORE_FEATURES)]
        model = cls(F.mean().to_dict(), F.std(ddof=0).to_dict(), 1.0, dict(DIRECTIONS))
        model.std_s = float(model.oriented(F).mean(axis=1).std(ddof=0))
        return model

    def oriented(self, F: pd.DataFrame) -> pd.DataFrame:
        """Clipped z-scores where larger always means better."""
        z = pd.DataFrame({name: self.directions[name] * (F[name] - self.mean[name]) / self.std[name]
                          for name in SCORE_FEATURES}, index=F.index)
        return z.clip(-CLIP, CLIP)

    def raw(self, F: pd.DataFrame) -> pd.Series:
        """s: mean oriented z (used for ranking and AUC)."""
        return self.oriented(F).mean(axis=1)

    def score(self, F: pd.DataFrame) -> pd.Series:
        return pd.Series(100.0 * _phi(self.raw(F) / self.std_s), index=F.index, name="score")

    def breakdown(self, F: pd.DataFrame) -> pd.DataFrame:
        """Score plus 0-100 sub-scores per group, for explaining one episode."""
        z = self.oriented(F)
        out = pd.DataFrame({f"sub_{g}": 100.0 * _phi(z[list(cols)].mean(axis=1))
                            for g, cols in GROUPS.items()}, index=F.index)
        return out.assign(score=self.score(F))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PARAMS) -> "QualityScore":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def score_episodes(episodes: list[Episode], model: QualityScore) -> pd.DataFrame:
    """One row per episode: status, raw inputs, sub-scores and score (never raises per file)."""
    rows = []
    for e in episodes:
        row = {"episode_id": e.episode_id, "label": e.label, "status": "ok"}
        if e.summary["issues"]:
            row["status"] = f"invalid: {e.summary['issues']}"
        else:
            try:
                row.update(score_inputs(e))
            except ValueError as error:
                if "never closes" not in str(error):
                    raise
                row["status"] = "no_grasp"
        rows.append(row)
    table = pd.DataFrame(rows)
    for name in SCORE_FEATURES:
        if name not in table:
            table[name] = np.nan
    ok = table["status"] == "ok"
    if ok.any():
        table = table.join(model.breakdown(table.loc[ok, list(SCORE_FEATURES)]))
    else:
        table = table.assign(sub_carry=np.nan, sub_smooth=np.nan, sub_exit=np.nan, score=np.nan)
    table.loc[table["status"] == "no_grasp", "score"] = 0.0
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="Score episodes 0-100 with the fixed formula.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="folder of CSVs, or a folder with success/ and fail/")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--output", type=Path, default=Path("reports/score/scores.csv"))
    parser.add_argument("--fit", action="store_true",
                        help="recompute reference statistics from --data-dir and save to --params")
    args = parser.parse_args()

    episodes = load_folder(args.data_dir)
    if args.fit:
        valid = [e for e in episodes if not e.summary["issues"]]
        model = QualityScore.fit(pd.DataFrame([score_inputs(e) for e in valid]))
        print(f"Saved reference statistics from {len(valid)} episodes: {model.save(args.params)}")
    else:
        model = QualityScore.load(args.params)

    table = score_episodes(episodes, model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.output, index=False)
    pd.set_option("display.width", 160)
    shown = ["episode_id", "status", "sub_carry", "sub_smooth", "sub_exit", "score"]
    print(table[shown].round(1).to_string(index=False))
    if (table["label"] >= 0).any():
        print("\nmean score by label (1 = success):")
        print(table[table["label"] >= 0].groupby("label")["score"].agg(["mean", "count"]).round(1))
    print(f"\nScores: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
