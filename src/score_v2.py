"""Quality score v2 (task 4) and 5th classifier (task 2): carry + smoothness + release.

    z_i   = d_i * (x_i - mean_i) / std_i, clipped to [-3, 3]
    s     = (z_1 + z_2 + z_3 + z_4) / 4                    (equal weights)
    score = 100 * Phi(s / std_s)          success if score >= 50 (reference average)

    i  input              d   sub-score  meaning
    1  delta_hold_l_sh_r  -1  carry      shoulder roll swings inward while carrying
    2  delta_hold_l_el_p  +1  carry      elbow extends toward the box (lower release in video)
    3  rev_elp_hold       -1  smooth     elbow direction reversals while carrying (hesitation)
    4  thumb_open_02s     -1  release    thumb opening in the first 0.2 s after release

Change from PR #4 (src/score.py): input 4 t_retract is replaced by thumb_open_02s.
- t_retract is measured after the bottle is released, so it cannot cause the fall, and it
  separates success/fail only in episodes 33-48.
- thumb_open_02s targets the failure type seen in the videos where the bottle stood on the
  box floor but tipped over while the hand opened (src/features_release.py).
The release input was chosen after looking at the videos and labels, so cross-validation on
the 48 reference episodes is optimistic for v2 (reports/fifth_model/README.md).

mean/std/std_s come from the 48 reference episodes (labels are not used) and are stored in
reports/score_v2/params.json, so the same formula scores any new CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .classify import DEFAULT_THRESHOLD, EqualWeightClassifier
from .data import Episode, load_dataset
from .features_phase import episode_features
from .features_release import release_features

DIRECTIONS = {
    "delta_hold_l_sh_r": -1.0,
    "delta_hold_l_el_p": 1.0,
    "rev_elp_hold": -1.0,
    "thumb_open_02s": -1.0,
}
SCORE_FEATURES = tuple(DIRECTIONS)
GROUPS = {"carry": ("delta_hold_l_sh_r", "delta_hold_l_el_p"),
          "smooth": ("rev_elp_hold",), "release": ("thumb_open_02s",)}
DEFAULT_PARAMS = Path(__file__).resolve().parents[1] / "reports/score_v2/params.json"


def score_inputs_v2(episode: Episode) -> dict[str, float]:
    """The four raw v2 inputs of one valid episode (raises ValueError if no grasp)."""
    values = {**episode_features(episode), **release_features(episode)}
    return {name: float(values[name]) for name in SCORE_FEATURES}


def fit_reference(episodes: list[Episode], threshold: float = DEFAULT_THRESHOLD) -> EqualWeightClassifier:
    """Reference statistics from the raw inputs of the reference episodes (labels unused)."""
    F = pd.DataFrame([score_inputs_v2(e) for e in episodes])
    return EqualWeightClassifier(directions=dict(DIRECTIONS), threshold=threshold, groups=dict(GROUPS)).fit(F)


def load(path: str | Path = DEFAULT_PARAMS) -> EqualWeightClassifier:
    return EqualWeightClassifier.load(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit the v2 reference statistics (normally not needed).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    args = parser.parse_args()
    model = fit_reference(load_dataset(args.data_dir))
    print(f"Saved v2 reference statistics: {model.save(args.params)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
