"""5th model / score v2 / cluster v2: add the release moment, compare with v1.

Classifiers (all: success if score >= 50 on equal-weight clipped z-scores)
  4th       inputs re-chosen in each training fold from 13 phase inputs (honest, v1 rule)
  v1 fixed  PR #4 inputs (sh_r, el_p, rev, t_retract), chosen on all 48 -> optimistic
  v2 fixed  sh_r, el_p, rev, thumb_open_02s, chosen with video + labels -> optimistic
  v1+thumb  PR #4 inputs + thumb_open_02s -> optimistic
  blind     in each training fold: 3 inputs from the 13 phase inputs + 1 from 28 release-
            window candidates, without any video knowledge -> what the data alone finds
Protocols: A = StratifiedKFold(5, shuffle) x seeds 0-9; B = 6 recording blocks of 8.
If reports/video_evidence/video_annotations.csv exists, scores are also checked per failure
type (A: released high/tilted, B: tipped while the hand opened) and the clusters are
cross-tabulated with those types.

Outputs: reports/fifth_model/comparison.json, cluster_v2_assignments.csv,
         release_signal.png, cluster_v2_map.png
Run from the repo root: python -m experiments.fifth_model_release
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, balanced_accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold

from src import score_v2
from src.classify import EqualWeightClassifier, select_score_inputs
from src.cluster import CLUSTER_INPUTS, PhaseClusterer, choose_k
from src.cluster_v2 import NAMES as V2_NAMES, MechanismClusterer
from src.data import load_dataset
from src.features_phase import extract_phase_features
from src.features_release import release_features
from src.phases import ARM, FPS, grip_signal, segment, state_series
from src.score import DIRECTIONS as V1_DIRECTIONS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/fifth_model"
ANNOTATIONS = ROOT / "reports/video_evidence/video_annotations.csv"
N_SEEDS, N_BLOCKS = 10, 6


def release_window_candidates(episode) -> dict[str, float]:
    """28 generic features of the first 0.1-0.3 s after the hand starts opening (blind pool)."""
    p = segment(episode)
    S = {j: state_series(episode, j) for j in ARM}
    out = {}
    for w in (3, 6, 9):
        b = min(p.n_frames - 1, p.open_start + w)
        delta = {j: S[j][b] - S[j][p.open_start] for j in ARM}
        out.update({f"d{w}_{j}": v for j, v in delta.items()})
        out[f"speed{w}"] = float(np.sqrt(sum(v ** 2 for v in delta.values())) / (w / FPS))
    g = grip_signal(episode)
    out["open_frames"] = float(p.open_end - p.open_start)
    out["finger_drop_2f"] = float(g[p.open_start] - g[min(p.n_frames - 1, p.open_start + 2)])
    a = max(0, p.open_start - 6)
    out["speed_pre_02s"] = float(np.sqrt(sum((S[j][p.open_start] - S[j][a]) ** 2 for j in ARM)) / 0.2)
    out.update(release_features(episode))
    return out


def equal_weight(F, directions=None, k=4, pool=None, release_pool=None):
    """fit_predict for an equal-weight rule; directions=None -> chosen in the training fold."""
    def fit_predict(tr, te, y):
        if directions is not None:
            dirs = dict(directions)
        elif release_pool is None:
            dirs = select_score_inputs(F.iloc[tr][pool], y[tr], k)
        else:
            dirs = select_score_inputs(F.iloc[tr][pool], y[tr], k - 1)
            extra = select_score_inputs(F.iloc[tr][release_pool], y[tr], 1)
            fit_predict.release_picks.append(next(iter(extra)))
            dirs.update(extra)
        model = EqualWeightClassifier(directions=dirs).fit(F.iloc[tr])
        return model.decision_function(F.iloc[te]).to_numpy()
    fit_predict.release_picks = []
    return fit_predict


def protocols(fit_predict, y, number):
    acc, bal, auc = [], [], []
    for seed in range(N_SEEDS):
        s = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
            s[te] = fit_predict(tr, te, y)
        acc.append(accuracy_score(y, s >= 0)); bal.append(balanced_accuracy_score(y, s >= 0)); auc.append(roc_auc_score(y, s))
    block = np.empty(len(y), dtype=int)
    block[np.argsort(number)] = np.arange(len(y)) * N_BLOCKS // len(y)
    s = np.zeros(len(y))
    for b in range(N_BLOCKS):
        s[block == b] = fit_predict(np.where(block != b)[0], np.where(block == b)[0], y)
    return {"A_accuracy_mean": float(np.mean(acc)), "A_accuracy_std": float(np.std(acc)),
            "A_balanced_accuracy_mean": float(np.mean(bal)), "A_roc_auc_mean": float(np.mean(auc)),
            "B_accuracy": float(accuracy_score(y, s >= 0)), "B_balanced_accuracy": float(balanced_accuracy_score(y, s >= 0)),
            "B_roc_auc": float(roc_auc_score(y, s))}


def eta2(groups, x):
    x = np.asarray(x, dtype=float)
    within = sum(((x[groups == g] - x[groups == g].mean()) ** 2).sum() for g in np.unique(groups))
    return float(1 - within / ((x - x.mean()) ** 2).sum())


def main():
    episodes = load_dataset(ROOT / "data")
    X, y_series, meta = extract_phase_features(episodes)
    y = y_series.to_numpy()
    number = meta["episode_id"].str.extract(r"(\d+)")[0].astype(int).to_numpy()
    R = pd.DataFrame([release_window_candidates(e) for e in episodes])
    F = pd.concat([X, meta[["t_reach", "t_hold", "t_retract"]], R], axis=1)
    pool = list(CLUSTER_INPUTS)
    v2_dirs = dict(score_v2.DIRECTIONS)
    plus = {**V1_DIRECTIONS, "thumb_open_02s": -1.0}

    blind = equal_weight(F, pool=pool, release_pool=list(R.columns))
    models = {
        "4th: v1 rule, inputs re-chosen in each training fold": equal_weight(F, pool=pool),
        "v1 fixed: PR #4 inputs (optimistic)": equal_weight(F, directions=V1_DIRECTIONS),
        "v2 fixed: sh_r, el_p, rev, thumb_open (optimistic, video-informed)": equal_weight(F, directions=v2_dirs),
        "v1 + thumb_open fixed (optimistic, video-informed)": equal_weight(F, directions=plus),
        "blind: 3 phase inputs + 1 of 28 release-window inputs chosen in-fold": blind,
    }
    result = {"classifiers": {}}
    for name, fp in models.items():
        result["classifiers"][name] = protocols(fp, y, number)
        r = result["classifiers"][name]
        print(f"{name:72s} A acc {r['A_accuracy_mean']:.3f}+-{r['A_accuracy_std']:.3f} auc {r['A_roc_auc_mean']:.3f}"
              f" | B acc {r['B_accuracy']:.3f} auc {r['B_roc_auc']:.3f}", flush=True)
    picks = Counter(blind.release_picks)
    result["blind_release_picks"] = {k: round(v / sum(picks.values()), 3) for k, v in picks.most_common(5)}

    # full-data scores of v1 and v2 (fixed reference statistics) for per-period / per-type checks
    v1 = EqualWeightClassifier(directions=V1_DIRECTIONS).fit(F)
    v2 = score_v2.load()
    s1, s2 = v1.score(F).to_numpy(), v2.score(F[list(score_v2.SCORE_FEATURES)]).to_numpy()
    third = np.digitize(number, [17, 33])
    result["full_data"] = {}
    for name, s in (("v1", s1), ("v2", s2)):
        result["full_data"][name] = {
            "accuracy_score_ge_50": float(((s >= 50).astype(int) == y).mean()),
            "roc_auc": float(roc_auc_score(y, s)),
            "roc_auc_by_third(1-16,17-32,33-48)": [round(float(roc_auc_score(y[third == b], s[third == b])), 3) for b in range(3)],
            "spearman_with_episode_number": round(float(spearmanr(s, number).correlation), 3),
            "mean_score_success": float(s[y == 1].mean()), "mean_score_fail": float(s[y == 0].mean())}

    # clusters: PR #3 vs v2
    Fc = F[list(CLUSTER_INPUTS)]
    c_v1 = PhaseClusterer(k=3).fit(Fc).predict(Fc)
    Z = v2.group_z(F[list(score_v2.SCORE_FEATURES)])
    mech = MechanismClusterer().fit(Z)
    c_v2 = mech.predict(Z)
    rng = np.random.default_rng(0)
    boots = [adjusted_rand_score(c_v2, KMeans(3, n_init=10, random_state=b).fit(Z.to_numpy()[rng.choice(len(Z), len(Z))]).predict(Z.to_numpy()))
             for b in range(50)]
    v1_sel = choose_k(Fc, ks=[3])
    result["clusters"] = {
        "v1_PR3": {"bootstrap_ari": float(v1_sel["bootstrap_ari"].iloc[0]), "silhouette": float(v1_sel["silhouette"].iloc[0]),
                   "eta2_episode_number": eta2(c_v1, number),
                   "success_rate": {int(c): round(float(y[c_v1 == c].mean()), 2) for c in range(3)},
                   "size": {int(c): int((c_v1 == c).sum()) for c in range(3)}},
        "v2_mechanism": {"bootstrap_ari": float(np.mean(boots)), "silhouette": float(silhouette_score(Z, c_v2)),
                         "eta2_episode_number": eta2(c_v2, number),
                         "success_rate": {int(c): round(float(y[c_v2 == c].mean()), 2) for c in range(3)},
                         "size": {int(c): int((c_v2 == c).sum()) for c in range(3)},
                         "names": {int(c): V2_NAMES[c] for c in range(3)},
                         "centres_z": mech.centres_.round(3).to_dict(orient="index")}}

    assignments = pd.DataFrame({"episode_id": meta["episode_id"], "label": y, "score_v1": s1.round(2), "score_v2": s2.round(2),
                                "cluster_v1": c_v1, "cluster_v2": c_v2, "cluster_v2_name": [V2_NAMES[c] for c in c_v2]})
    assignments = assignments.join(Z.round(3).add_prefix("z_"))

    if ANNOTATIONS.exists():
        ann = pd.read_csv(ANNOTATIONS)
        kind = pd.Series(np.where(y == 1, "S", ""), index=number)
        kind.loc[ann.loc[ann.fail_type.notna(), "ep"].to_numpy()] = ann.loc[ann.fail_type.notna(), "fail_type"].to_numpy()
        kind = kind.loc[number].to_numpy()
        assignments["video_type"] = kind
        early_s, late_s = (y == 1) & (number <= 28), (y == 1) & (number >= 19)
        result["per_failure_type_auc"] = {}
        for name, s in (("v1", s1), ("v2", s2)):
            a = (kind == "A") | early_s
            b = (kind == "B") | late_s
            result["per_failure_type_auc"][name] = {
                "success(ep<=28)_vs_typeA": round(float(roc_auc_score(y[a], s[a])), 3),
                "success(ep>=19)_vs_typeB": round(float(roc_auc_score(y[b], s[b])), 3)}
        for key, c in (("v1_PR3", c_v1), ("v2_mechanism", c_v2)):
            result["clusters"][key]["crosstab_video_type"] = pd.crosstab(c, kind).to_dict(orient="index")
            result["clusters"][key]["ari_with_video_type"] = round(float(adjusted_rand_score(kind, c)), 3)
        # figure 1: thumb yaw aligned at release (log signal of failure type B)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        t = np.arange(-15, 31)
        colors = {"S": "#1f77b4", "A": "#ff7f0e", "B": "#d62728"}
        labels = {"S": "success", "A": "fail A: released high/tilted", "B": "fail B: tipped while hand opened"}
        for k in ("S", "A", "B"):
            traces = []
            for e, kk in zip(episodes, kind):
                if kk != k:
                    continue
                p = segment(e); th = state_series(e, "l_hand_thumb_y")
                idx = np.clip(p.open_start + t, 0, p.n_frames - 1)
                traces.append(th[idx] - th[p.open_start])
            m = np.mean(traces, axis=0)
            axes[0].plot(t / FPS, m, color=colors[k], lw=2, label=f"{labels[k]} (n={len(traces)})")
            axes[0].fill_between(t / FPS, np.percentile(traces, 25, axis=0), np.percentile(traces, 75, axis=0), color=colors[k], alpha=0.15)
        axes[0].axvline(0, color="k", ls="--", lw=1); axes[0].axvline(0.2, color="gray", ls=":", lw=1)
        axes[0].set(title="Thumb yaw around release (0 = fingers start opening)", xlabel="time from release (s)",
                    ylabel="thumb yaw change (rad)")
        axes[0].legend(fontsize=7)
        for k in ("S", "A", "B"):
            m = kind == k
            axes[1].scatter(Z["carry"][m], Z["release"][m], c=colors[k], s=28, label=labels[k])
        for c, (cx, cy) in mech.centres_[["carry", "release"]].iterrows():
            axes[1].annotate(f"C{c}", (cx, cy), fontsize=11, weight="bold")
        axes[1].axhline(0, color="gray", lw=0.6); axes[1].axvline(0, color="gray", lw=0.6)
        axes[1].set(title="v2 sub-scores (z): carry vs release, C = cluster centre", xlabel="carry (larger = better)",
                    ylabel="release (larger = gentler)")
        axes[1].legend(fontsize=7)
        fig.tight_layout(); OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / "release_signal.png", dpi=150); plt.close(fig)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    assignments.to_csv(OUT / "cluster_v2_assignments.csv", index=False)
    print(json.dumps({k: result[k] for k in ("full_data", "clusters", "blind_release_picks")}, indent=1)[:4000])
    if "per_failure_type_auc" in result:
        print(json.dumps(result["per_failure_type_auc"], indent=1))


if __name__ == "__main__":
    main()
