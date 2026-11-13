"""Fixed, label-independent episode summaries of the 52 motion channels."""

import numpy as np
import pandas as pd

from .data import Episode, MOTION_COLUMNS


def extract_features(episodes: list[Episode]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X, y and metadata separately; no IDs or duration in X.

    Standard deviation uses ddof=0. Delta means final minus initial value.
    """
    features, metadata = [], []
    for episode in episodes:
        values = np.asarray(episode.motion_inputs(), dtype=float)
        statistics = {
            "mean": values.mean(axis=0), "std": values.std(axis=0, ddof=0),
            "range": np.ptp(values, axis=0), "delta": values[-1] - values[0],
        }
        features.append({f"{column}__{stat}": float(vector[i])
                         for stat, vector in statistics.items()
                         for i, column in enumerate(MOTION_COLUMNS)})
        metadata.append({"episode_id": episode.episode_id, "path": str(episode.path)})
    return (pd.DataFrame(features), pd.Series([e.label for e in episodes], name="label"),
            pd.DataFrame(metadata))
