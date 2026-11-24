"""Release-moment feature of the left hand (5th model, quality score v2).

thumb_open_02s
    Decrease of the thumb yaw state during the first 0.2 s (6 frames) after the four
    fingers start opening:  thumb_y[open_start] - thumb_y[open_start + 6]  (rad).
    Larger = the thumb swings open faster = more abrupt release.

Where it comes from: the 1st-dataset videos are frame-aligned with the CSV logs. They
showed a failure type in which the bottle already stood on the box floor and tipped over
while the hand opened (episodes 19, 21, 25, 35, 44, 45, 48; reports/video_evidence/).
Episode-level features before this one summarize the motion only up to the release, so
they cannot see this moment.

Caution: the feature was picked after looking at those videos and the labels, so its gain
on these 48 episodes is optimistic (reports/fifth_model/README.md). Prediction uses only
the CSV log; videos are never a model input.
"""

from __future__ import annotations

from .data import Episode
from .phases import segment, state_series

WINDOW = 6  # frames = 0.2 s at 30 fps


def release_features(episode: Episode) -> dict[str, float]:
    """Release features of one valid episode (raises ValueError if the hand never closes)."""
    phases = segment(episode)
    thumb = state_series(episode, "l_hand_thumb_y")
    end = min(phases.n_frames - 1, phases.open_start + WINDOW)
    return {"thumb_open_02s": float(thumb[phases.open_start] - thumb[end])}
