"""4th model: use the quality score as the classifier, compared with the 3rd model and with
more complex models under two validation protocols.

Protocols (48 episodes)
  A. StratifiedKFold(5, shuffle) x seeds 0-9: accuracy, balanced accuracy, ROC AUC (mean, std)
  B. Recording blocks: episodes sorted by number, 6 blocks of 8, leave one block out.
     Success rate rises with episode number, so B checks whether a model still works on
     episodes recorded at another time (closer to unseen demo data than A).
Everything that looks at labels (input selection, directions, weights, thresholds) is fitted
on the training folds only, except the 'fixed inputs' row, which reuses the four inputs that
were chosen on all 48 episodes and is therefore optimistic.

Outputs: reports/fourth_model/comparison.json, oof_predictions.csv
Run from the repo root: python -m experiments.fourth_model_score_classifier
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.classify import EqualWeightClassifier, select_score_inputs
from src.cluster import CLUSTER_INPUTS
from src.data import load_dataset
from src.features_phase import FEATURE_NAMES, extract_phase_features
from src.phases import ARM, state_series
from src.score import DIRECTIONS

ROOT = Path(__file__).resolve().parents[1]
N_SEEDS = 10
N_BLOCKS = 6
POOL = list(CLUSTER_INPUTS)          # 10 phase features + t_reach, t_hold, t_retract


# ---------------------------------------------------------------- models
# Each model is fit_predict(train_idx, test_idx) -> (decision values, 0/1 predictions).

def sklearn_model(make, X):
    X = np.asarray(X, dtype=float)

    def fit_predict(tr, te, y):
        model = make().fit(X[tr], y[tr])
        if hasattr(model, "predict_proba"):
            decision = model.predict_proba(X[te])[:, 1]
        else:
            decision = model.decision_function(X[te])
        return decision, model.predict(X[te])
    return fit_predict


def score_rule(F, directions=None, tune_threshold=False):
    def fit_predict(tr, te, y):
        model = EqualWeightClassifier(directions=directions).fit(F.iloc[tr], y[tr])
        if tune_threshold:  # best balanced accuracy on the training fold (for comparison only)
            train_scores = model.score(F.iloc[tr]).to_numpy()
            model.threshold = max(np.unique(train_scores), key=lambda t: balanced_accuracy_score(
                y[tr], (train_scores >= t).astype(int)))
        return model.decision_function(F.iloc[te]).to_numpy(), model.predict(F.iloc[te])
    return fit_predict


def trajectories(episodes, n_points=40):
    """Left arm 7 joints + thumb pitch, each resampled to n_points on normalized time."""
    rows = []
    for e in episodes:
        channels = [state_series(e, j) for j in ARM + ("l_hand_thumb_p",)]
        grid = np.linspace(0, len(channels[0]) - 1, n_points)
        rows.append(np.concatenate([np.interp(grid, np.arange(len(c)), c) for c in channels]))
    return np.array(rows)


def rocket_features(episodes, n_kernels=2000, seed=0):
    """ROCKET (Dempster et al., 2020): random dilated convolution kernels, features = proportion
    of positive values and maximum of each convolution. The kernels do not depend on the data,
    so transforming all episodes once does not leak labels. Channels are z-scored globally."""
    series = [np.vstack([state_series(e, j) for j in ARM + ("l_hand_thumb_p",)]) for e in episodes]
    stacked = np.concatenate(series, axis=1)
    mean, std = stacked.mean(axis=1, keepdims=True), stacked.std(axis=1, keepdims=True)
    series = [(s - mean) / std for s in series]
    n_channels, min_length = series[0].shape[0], min(s.shape[1] for s in series)
    rng = np.random.default_rng(seed)
    kernels = []
    for _ in range(n_kernels):
        length = int(rng.choice([7, 9, 11]))
        n_used = int(min(n_channels, 2 ** rng.uniform(0, np.log2(n_channels + 1))))
        weights = rng.normal(size=(n_used, length))
        weights -= weights.mean(axis=1, keepdims=True)
        dilation = int(2 ** rng.uniform(0, np.log2((min_length - 1) / (length - 1))))
        padding = ((length - 1) * dilation) // 2 if rng.integers(2) else 0
        kernels.append((rng.choice(n_channels, n_used, replace=False), weights,
                        rng.uniform(-1, 1), dilation, padding, length))
    out = np.zeros((len(series), 2 * n_kernels))
    for i, x in enumerate(series):
        for k, (channels, weights, bias, dilation, padding, length) in enumerate(kernels):
            xp = np.pad(x[channels], ((0, 0), (padding, padding)))
            n_out = xp.shape[1] - (length - 1) * dilation
            index = np.arange(n_out)[:, None] + np.arange(length)[None, :] * dilation
            conv = np.einsum("cij,cj->i", xp[:, index], weights) + bias
            out[i, 2 * k], out[i, 2 * k + 1] = (conv > 0).mean(), conv.max()
    return out


# ---------------------------------------------------------------- protocols

def protocol_a(fit_predict, y):
    accuracy, balanced, auc = [], [], []
    wrong = np.zeros(len(y))
    decision_sum = np.zeros(len(y))
    for seed in range(N_SEEDS):
        decision, prediction = np.zeros(len(y)), np.zeros(len(y), dtype=int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
            decision[te], prediction[te] = fit_predict(tr, te, y)
        accuracy.append(accuracy_score(y, prediction))
        balanced.append(balanced_accuracy_score(y, prediction))
        auc.append(roc_auc_score(y, decision))
        wrong += prediction != y
        decision_sum += decision
    summary = {"accuracy_mean": float(np.mean(accuracy)), "accuracy_std": float(np.std(accuracy)),
               "balanced_accuracy_mean": float(np.mean(balanced)), "roc_auc_mean": float(np.mean(auc)),
               "roc_auc_std": float(np.std(auc))}
    return summary, wrong / N_SEEDS, decision_sum / N_SEEDS


def protocol_b(fit_predict, y, number):
    block = np.empty(len(y), dtype=int)
    block[np.argsort(number)] = np.arange(len(y)) * N_BLOCKS // len(y)
    decision, prediction = np.zeros(len(y)), np.zeros(len(y), dtype=int)
    for b in range(N_BLOCKS):
        tr, te = np.where(block != b)[0], np.where(block == b)[0]
        decision[te], prediction[te] = fit_predict(tr, te, y)
    return {"accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "roc_auc": float(roc_auc_score(y, decision))}


def main():
    episodes = load_dataset(ROOT / "data")
    X, y_series, metadata = extract_phase_features(episodes)
    y = y_series.to_numpy()
    number = metadata["episode_id"].str.extract(r"(\d+)")[0].astype(int).to_numpy()
    F = pd.concat([X, metadata[["t_reach", "t_hold", "t_retract"]]], axis=1)[POOL]
    phase10 = X[list(FEATURE_NAMES)]

    def logistic(C):
        return lambda: make_pipeline(VarianceThreshold(), StandardScaler(),
                                     LogisticRegression(C=C, max_iter=5000))

    models = {
        "3rd: phase features 10 + LogReg C=1": sklearn_model(logistic(1.0), phase10),
        "phase features 10 + LogReg C=0.1": sklearn_model(logistic(0.1), phase10),
        "4th: score >= 50, inputs re-chosen in each training fold": score_rule(F),
        "score >= 50, PR #4 inputs fixed (optimistic)": score_rule(F, directions=dict(DIRECTIONS)),
        "score, threshold tuned on training fold": score_rule(F, tune_threshold=True),
        "trajectory 8x40 + LogReg C=0.1": sklearn_model(
            lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)), trajectories(episodes)),
        "ROCKET 2000 kernels + ridge": sklearn_model(
            lambda: make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-1, 4, 12))),
            rocket_features(episodes)),
    }

    result, wrong_rates, chosen_decision = {}, {}, None
    for name, fit_predict in models.items():
        a, wrong, decision = protocol_a(fit_predict, y)
        result[name] = {"protocol_a_repeated_cv": a, "protocol_b_recording_blocks": protocol_b(fit_predict, y, number)}
        wrong_rates[name] = wrong
        if name.startswith("4th"):
            chosen_decision = decision
        print(f"{name:58s} A: acc {a['accuracy_mean']:.3f}+-{a['accuracy_std']:.3f} "
              f"auc {a['roc_auc_mean']:.3f} | B: acc {result[name]['protocol_b_recording_blocks']['accuracy']:.3f} "
              f"auc {result[name]['protocol_b_recording_blocks']['roc_auc']:.3f}", flush=True)

    # how stable is the input selection of the 4th model?
    picks = Counter()
    for seed in range(N_SEEDS):
        for tr, _ in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
            picks[tuple(sorted(select_score_inputs(F.iloc[tr], y[tr])))] += 1
    total = sum(picks.values())
    result["input_sets_chosen_in_training_folds"] = [
        {"inputs": list(s), "share": round(c / total, 3)} for s, c in picks.most_common(5)]
    single = Counter(name for s, c in picks.items() for name in s for _ in range(c))
    result["input_frequency_in_training_folds"] = {name: round(c / total, 3) for name, c in single.most_common()}

    hard = {name: sorted(number[(y == 0) & (w >= 0.8)].tolist()) for name, w in wrong_rates.items()}
    result["fails_wrong_in_80pct_of_seeds"] = hard
    votes = Counter(e for episodes_missed in hard.values() for e in episodes_missed)
    result["fails_missed_by_most_models"] = sorted(e for e, v in votes.items() if v > len(hard) / 2)
    result["protocol"] = ("A: StratifiedKFold(5, shuffle) x seeds 0-9; B: 6 recording blocks of 8 episodes, "
                          "leave one block out; selection/threshold fitted on training folds only")
    result["majority_baseline_accuracy"] = float(max(y.mean(), 1 - y.mean()))

    out = ROOT / "reports/fourth_model"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({"episode_id": metadata["episode_id"], "label": y,
                  "mean_decision_4th": chosen_decision.round(4),
                  "wrong_rate_4th": wrong_rates["4th: score >= 50, inputs re-chosen in each training fold"],
                  "wrong_rate_3rd": wrong_rates["3rd: phase features 10 + LogReg C=1"]}).to_csv(
        out / "oof_predictions.csv", index=False)
    print("\ninput sets chosen in training folds:", result["input_sets_chosen_in_training_folds"][:3])
    print("fails missed by most models:", result["fails_missed_by_most_models"])


if __name__ == "__main__":
    main()
