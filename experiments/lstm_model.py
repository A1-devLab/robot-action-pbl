"""LSTM sequence classifier, compared under the same protocols as the 4th model.

Input: left arm 7 joints + thumb pitch, each episode resampled to 40 (or 20) steps on normalized
time (the same 8x40 trajectories as the 'trajectory LogReg' row), shaped (episode, step, channel).
Rows: the first try (h=16, last hidden state) overfits; the regularized rows add mean pooling
over time, L2 1e-2, dropout 0.5, 8 augmented copies, 20 steps and a 5-seed average.
Standardization and augmentation (time warp, amplitude scale, noise) happen inside fit(),
so only training folds are used and augmented. Protocols A and B are imported unchanged.

Outputs: reports/lstm/comparison.json
Run from the repo root: python -m experiments.lstm_model
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.fourth_model_score_classifier import (
    protocol_a, protocol_b, sklearn_model, trajectories)
from src.data import load_dataset
from src.features_phase import extract_phase_features
from src.lstm import LSTMClassifier

ROOT = Path(__file__).resolve().parents[1]
N_POINTS, N_CHANNELS = 40, 8


def lstm_model(sequences, n_seeds=1, **params):
    """n_seeds > 1 averages the probabilities of LSTMs trained with different seeds."""
    def fit_predict(tr, te, y):
        p = np.mean([LSTMClassifier(seed=s, **params).fit(sequences[tr], y[tr]).predict_proba(
            sequences[te])[:, 1] for s in range(n_seeds)], axis=0)
        return p, (p >= 0.5).astype(int)
    return fit_predict


def to_sequences(episodes, n_points):
    flat = trajectories(episodes, n_points)                        # (n, 8*n_points), channel-major
    return flat, flat.reshape(len(flat), N_CHANNELS, n_points).transpose(0, 2, 1)


def main():
    episodes = load_dataset(ROOT / "data")
    _, y_series, metadata = extract_phase_features(episodes)
    y = y_series.to_numpy()
    number = metadata["episode_id"].str.extract(r"(\d+)")[0].astype(int).to_numpy()
    flat, sequences = to_sequences(episodes, N_POINTS)
    _, short = to_sequences(episodes, 20)
    regularized = dict(hidden=8, l2=1e-2, dropout=0.5, n_augment=8, pooling="mean")

    models = {
        "trajectory 8x40 + LogReg C=0.1 (reference)": sklearn_model(
            lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)), flat),
        "LSTM first try: h=16, last state, augmentation x4": lstm_model(sequences, hidden=16, n_augment=4),
        "LSTM regularized: h=8, mean pooling, 40 steps": lstm_model(sequences, **regularized),
        "LSTM regularized: h=8, mean pooling, 20 steps": lstm_model(short, **regularized),
        "LSTM regularized, 20 steps, 5-seed average": lstm_model(short, n_seeds=5, **dict(regularized, n_augment=4)),
    }

    result = {}
    for name, fit_predict in models.items():
        start = time.perf_counter()
        a, _, _ = protocol_a(fit_predict, y)
        seconds_a = time.perf_counter() - start
        b = protocol_b(fit_predict, y, number)
        result[name] = {"protocol_a_repeated_cv": a, "protocol_b_recording_blocks": b,
                        "seconds_per_fit": round(seconds_a / 50, 3)}
        print(f"{name:52s} A: acc {a['accuracy_mean']:.3f}+-{a['accuracy_std']:.3f} "
              f"bal {a['balanced_accuracy_mean']:.3f} auc {a['roc_auc_mean']:.3f} | "
              f"B: acc {b['accuracy']:.3f} auc {b['roc_auc']:.3f} | {seconds_a / 50:.2f}s/fit", flush=True)

    # training fit on all 48: does the LSTM simply memorize?
    for name, data, params in [("first_try", sequences, dict(hidden=16, n_augment=4)),
                                ("regularized_20_steps", short, regularized)]:
        full = LSTMClassifier(**params).fit(data, y)
        result[f"train_accuracy_all_48_{name}"] = float((full.predict(data) == y).mean())
    result["input"] = f"{N_CHANNELS} channels (left arm 7 + thumb pitch) x {N_POINTS} steps, normalized time"
    result["protocol"] = "same as reports/fourth_model (A: 5-fold x 10 seeds, B: 6 recording blocks)"
    result["majority_baseline_accuracy"] = float(max(y.mean(), 1 - y.mean()))

    out = ROOT / "reports/lstm"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("train accuracy on all 48:", {k: v for k, v in result.items() if k.startswith("train_")})


if __name__ == "__main__":
    main()
