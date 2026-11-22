"""Compare 1st (208 stats), 2nd (left-arm stats 28) and 3rd (phase features 10) models.

Same protocol as before: src.train.evaluate, StratifiedKFold(5, shuffle, seed 42),
L2 LogisticRegression(C=1). Repeated over seeds 0-9 to show split variance.
"""

import json
from pathlib import Path

import numpy as np

from src.data import load_dataset
from src.features import extract_features
from src.features_phase import extract_phase_features
from src.train import evaluate


ROOT = Path(__file__).resolve().parents[1]
JOINTS = ("sh_p", "sh_r", "sh_y", "el_p", "wr_y", "wr_p", "wr_r")


def main():
    episodes = load_dataset(ROOT / "data")
    X_stats, y, metadata = extract_features(episodes)
    left_arm = [c for c in X_stats.columns if any(c.startswith(f"state.l_{j}__") for j in JOINTS)]
    X_phase, _, phase_meta = extract_phase_features(episodes)
    models = {"baseline_208": X_stats, "left_arm_state_28": X_stats[left_arm], "phase_10": X_phase}

    result, summary = {}, {}
    for name, X in models.items():
        metrics, predictions = evaluate(X, y, seed=42)
        repeated = [evaluate(X, y, seed=s)[0] for s in range(10)]
        acc = [m["accuracy"] for m in repeated]
        bal = [m["balanced_accuracy"] for m in repeated]
        result[name] = {"seed42": metrics, "seeds0_9": {
            "accuracy_mean": float(np.mean(acc)), "accuracy_std": float(np.std(acc)),
            "balanced_accuracy_mean": float(np.mean(bal)), "balanced_accuracy_std": float(np.std(bal))}}
        summary[name] = {"seed42_accuracy": round(metrics["accuracy"], 4),
                         "seed42_balanced": round(metrics["balanced_accuracy"], 4),
                         "seed42_confusion": metrics["confusion_matrix"],
                         "seeds0_9_accuracy": f"{np.mean(acc):.3f} +- {np.std(acc):.3f}"}
        if name == "phase_10":
            out = ROOT / "reports/third_model"
            out.mkdir(parents=True, exist_ok=True)
            metadata[["episode_id"]].join(predictions).to_csv(out / "phase_oof_predictions.csv", index=False)
            phase_meta.join(X_phase).join(y).to_csv(out / "phase_features.csv", index=False)

    result["protocol"] = ("src.train.evaluate; StratifiedKFold(5, shuffle=True); "
                          "L2 LogisticRegression(C=1); seed 42 and seeds 0-9")
    (ROOT / "reports/third_model/comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
