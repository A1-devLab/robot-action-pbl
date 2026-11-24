"""Task 2, 4th model: predict success/fail with the task-4 quality score.

Rule
    success  if  score >= threshold        (default 50 = average of the reference episodes)
    score    = src.score formula: equal-weight mean of clipped z-scores of four phase inputs,
               mapped to 0-100 with the standard normal CDF

Why this rule (experiments/fourth_model_score_classifier.py, reports/fourth_model/README.md):
- It ranks success above fail better than the 3rd model (phase features + logistic
  regression), also when the four inputs are re-chosen inside every training fold.
- Classification and quality score become one formula, so every prediction is explained
  by the same sub-scores (carry, smooth, exit) that the score reports.
- The threshold is fixed instead of tuned: tuning it on the training folds raised
  stratified-CV accuracy but lowered accuracy on held-out recording blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

DEFAULT_THRESHOLD = 50.0
CLIP = 3.0


def select_score_inputs(F: pd.DataFrame, y, k: int = 4, max_corr: float = 0.9) -> dict[str, float]:
    """Choose k inputs and their directions from candidate columns of F.

    Columns are ranked by |single-feature AUC - 0.5|; a column is skipped when its
    |Pearson r| with an already chosen column is >= max_corr. The direction is +1 when
    larger values go with success (AUC > 0.5) and -1 otherwise. This is the rule behind
    the four score inputs; experiments re-run it inside each training fold so that the
    reported cross-validation includes the selection step.
    """
    y = np.asarray(y)
    if len(np.unique(y)) != 2:
        raise ValueError("Input selection needs both classes in y")
    usable = [c for c in F.columns if F[c].std(ddof=0) > 0]
    auc = pd.Series({c: roc_auc_score(y, F[c]) for c in usable})
    ranked = (auc - 0.5).abs().sort_values(ascending=False, kind="mergesort").index
    corr = F[usable].corr().abs()
    chosen: list[str] = []
    for column in ranked:
        if all(corr.loc[column, other] < max_corr for other in chosen):
            chosen.append(column)
        if len(chosen) == k:
            break
    return {c: (1.0 if auc[c] > 0.5 else -1.0) for c in chosen}


@dataclass
class EqualWeightClassifier:
    """Equal-weight z-score rule with a fixed threshold on the 0-100 score scale.

    With directions = src.score.DIRECTIONS and the 48 reference episodes this reproduces
    src.score exactly. directions=None chooses inputs with select_score_inputs in fit
    (used only for honest cross-validation of the selection rule).
    """

    directions: dict[str, float] | None = None
    k: int = 4
    max_corr: float = 0.9
    threshold: float = DEFAULT_THRESHOLD

    def fit(self, F: pd.DataFrame, y=None) -> "EqualWeightClassifier":
        if self.directions is None:
            if y is None:
                raise ValueError("y is required when directions are chosen from data")
            self.directions_ = select_score_inputs(F, y, self.k, self.max_corr)
        else:
            self.directions_ = dict(self.directions)
        columns = list(self.directions_)
        self.mean_ = F[columns].mean()
        self.std_ = F[columns].std(ddof=0).replace(0.0, 1.0)
        self.std_s_ = float(self.decision_function(F).std(ddof=0)) or 1.0
        return self

    def oriented(self, F: pd.DataFrame) -> pd.DataFrame:
        """Clipped z-scores where larger always means more success-like."""
        z = pd.DataFrame({c: d * (F[c] - self.mean_[c]) / self.std_[c]
                          for c, d in self.directions_.items()}, index=F.index)
        return z.clip(-CLIP, CLIP)

    def decision_function(self, F: pd.DataFrame) -> pd.Series:
        """s = mean oriented z; s >= 0 is the same as score >= 50."""
        return self.oriented(F).mean(axis=1)

    def score(self, F: pd.DataFrame) -> pd.Series:
        return pd.Series(100.0 * norm.cdf(self.decision_function(F) / self.std_s_),
                         index=F.index, name="score")

    def predict(self, F: pd.DataFrame) -> np.ndarray:
        return predict_from_score(self.score(F), self.threshold)


def predict_from_score(score, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """1 (success) where score >= threshold, else 0. NaN scores give 0; callers that keep
    invalid episodes should mask them separately."""
    values = np.asarray(score, dtype=float)
    return (np.nan_to_num(values, nan=-np.inf) >= threshold).astype(int)
