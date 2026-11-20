"""Run baseline and left-arm feature selection with the same sklearn protocol."""

import json
from pathlib import Path

from src.data import load_dataset
from src.features import extract_features
from src.train import evaluate


ROOT = Path(__file__).resolve().parents[1]


def main():
    episodes = load_dataset(ROOT / "data")
    X, y, metadata = extract_features(episodes)
    joints = ("sh_p", "sh_r", "sh_y", "el_p", "wr_y", "wr_p", "wr_r")
    selected = [c for c in X.columns if any(c.startswith(f"state.l_{joint}__") for joint in joints)]
    baseline_metrics, _ = evaluate(X, y, seed=42)
    second_metrics, predictions = evaluate(X[selected], y, seed=42)
    out = ROOT / "reports/second_model"
    out.mkdir(parents=True, exist_ok=True)
    result = {"baseline_sklearn": baseline_metrics,
              "left_arm_state_sklearn": second_metrics,
              "selected_features": selected,
              "protocol": "src.train.evaluate; StratifiedKFold(5, shuffle=True, random_state=42); L2 LogisticRegression(C=1)"}
    (out / "sklearn_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    metadata[["episode_id"]].join(predictions).to_csv(out / "sklearn_oof_predictions.csv", index=False)
    print(json.dumps({"baseline_accuracy": baseline_metrics["accuracy"],
                      "second_accuracy": second_metrics["accuracy"],
                      "baseline_balanced_accuracy": baseline_metrics["balanced_accuracy"],
                      "second_balanced_accuracy": second_metrics["balanced_accuracy"],
                      "baseline_confusion_matrix": baseline_metrics["confusion_matrix"],
                      "second_confusion_matrix": second_metrics["confusion_matrix"]}, indent=2))


if __name__ == "__main__":
    main()
