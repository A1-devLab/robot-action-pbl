"""Ten interpretable phase features of the left arm (no IDs, timing or right arm).

Definitions (S = joint state, v = joint velocity after 5-frame moving average):
    pose_grasp_l_sh_r     S(l_sh_r) at close_start       shoulder roll when grasping
    pose_grasp_l_wr_p     S(l_wr_p) at close_start       wrist pitch when grasping
    pose_grasp_l_wr_r     S(l_wr_r) at close_start       wrist roll when grasping
    speed_at_grasp        mean |v|_2 over close_start +-3 frames (moving while grasping?)
    delta_hold_l_sh_r     S(l_sh_r) at open_start - at close_end   shoulder roll moved while carrying
    delta_hold_l_el_p     same for elbow pitch                       elbow extension while carrying
    pauses_hold           runs of |v|_2 < 0.12 rad/s lasting >= 6 frames (0.2 s) during hold
    rev_elp_hold          sign changes of elbow velocity (9-frame average) during hold
    lower_amount          max S(l_sh_p) during hold - S(l_sh_p) at open_start   how far the arm is lowered
    pose_release_l_el_p   S(l_el_p) at open_start         elbow pose when releasing
Episode number, frame index, duration and right-arm channels are excluded on purpose:
they correlate with recording order (operator learning), not with task quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Episode
from .phases import ARM, FPS, segment, state_series

FEATURE_NAMES = (
    "pose_grasp_l_sh_r", "pose_grasp_l_wr_p", "pose_grasp_l_wr_r", "speed_at_grasp",
    "delta_hold_l_sh_r", "delta_hold_l_el_p", "pauses_hold", "rev_elp_hold",
    "lower_amount", "pose_release_l_el_p",
)
PAUSE_SPEED = 0.12   # rad/s
PAUSE_FRAMES = 6     # 0.2 s


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """Centered moving average with edge padding (no artificial spikes at the ends)."""
    padded = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def velocity(x: np.ndarray, k: int = 5) -> np.ndarray:
    return np.gradient(moving_average(x, k)) * FPS


def count_runs(mask: np.ndarray, min_length: int) -> int:
    count = length = 0
    for value in mask:
        length = length + 1 if value else 0
        count += length == min_length
    return count


def episode_features(episode: Episode) -> dict[str, float]:
    p = segment(episode)
    S = {j: state_series(episode, j) for j in ARM}
    speed = np.sqrt(sum(velocity(S[j]) ** 2 for j in ARM))
    hold = p.hold
    elbow_v = np.gradient(moving_average(S["l_el_p"], 9))[hold]
    signs = np.sign(elbow_v[elbow_v != 0])
    lo, hi = max(0, p.close_start - 3), min(p.n_frames, p.close_start + 4)
    sh_p_hold = S["l_sh_p"][p.close_end:p.open_start + 1]
    return {
        "pose_grasp_l_sh_r": S["l_sh_r"][p.close_start],
        "pose_grasp_l_wr_p": S["l_wr_p"][p.close_start],
        "pose_grasp_l_wr_r": S["l_wr_r"][p.close_start],
        "speed_at_grasp": float(speed[lo:hi].mean()),
        "delta_hold_l_sh_r": S["l_sh_r"][p.open_start] - S["l_sh_r"][p.close_end],
        "delta_hold_l_el_p": S["l_el_p"][p.open_start] - S["l_el_p"][p.close_end],
        "pauses_hold": float(count_runs(speed[hold] < PAUSE_SPEED, PAUSE_FRAMES)),
        "rev_elp_hold": float((signs[1:] != signs[:-1]).sum()),
        "lower_amount": float(sh_p_hold.max() - S["l_sh_p"][p.open_start]),
        "pose_release_l_el_p": S["l_el_p"][p.open_start],
    }


def extract_phase_features(episodes: list[Episode]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X (10 phase features), y and metadata, like features.extract_features."""
    rows, metadata = [], []
    for episode in episodes:
        rows.append(episode_features(episode))
        p = segment(episode)
        metadata.append({"episode_id": episode.episode_id, **{f"t_{k}": v for k, v in p.seconds().items()}})
    X = pd.DataFrame(rows, columns=list(FEATURE_NAMES)).astype(float)
    return X, pd.Series([e.label for e in episodes], name="label"), pd.DataFrame(metadata)
