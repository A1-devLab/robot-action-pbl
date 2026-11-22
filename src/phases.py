"""Split one pick-and-place episode into reach, grasp, hold, release and retract.

The left-hand finger state is nearly binary (open 0, closed about pi/2), so the
mean of the four finger joints g(t) marks when the bottle is grasped and released.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Episode

FPS = 30.0
ARM = ("l_sh_p", "l_sh_r", "l_sh_y", "l_el_p", "l_wr_y", "l_wr_p", "l_wr_r")
FINGERS = ("l_hand_index", "l_hand_middle", "l_hand_ring", "l_hand_pinky")
OPEN_BELOW = 0.3     # g(t) below this: hand open
CLOSED_ABOVE = 1.2   # g(t) above this: hand fully closed


@dataclass(frozen=True)
class Phases:
    """Frame indices of the four hand events (close_start < close_end <= open_start < open_end)."""

    close_start: int   # last open frame before the grasp
    close_end: int     # first fully closed frame (hold starts)
    open_start: int    # first frame after the closed run (release starts)
    open_end: int      # first open frame after the release
    n_frames: int

    @property
    def hold(self) -> slice:
        return slice(self.close_end, self.open_start)

    def seconds(self) -> dict[str, float]:
        """Phase durations in seconds."""
        return {
            "reach": self.close_start / FPS,
            "grasp": (self.close_end - self.close_start) / FPS,
            "hold": (self.open_start - self.close_end) / FPS,
            "release": (self.open_end - self.open_start) / FPS,
            "retract": (self.n_frames - self.open_end) / FPS,
        }


def state_series(episode: Episode, joint: str) -> np.ndarray:
    """State values of one joint over time (raises on invalid episodes)."""
    if episode.summary["issues"]:
        raise ValueError(f"Invalid episode {episode.episode_id}: {episode.summary['issues']}")
    return np.array([row[f"state.{joint}"] for row in episode.frames], dtype=float)


def grip_signal(episode: Episode) -> np.ndarray:
    """g(t): mean state of the four left-hand finger joints."""
    return np.mean([state_series(episode, f) for f in FINGERS], axis=0)


def segment(episode: Episode) -> Phases:
    """Return the hand events of the longest fully closed run.

    Raises ValueError if the hand never closes, so callers can handle such
    episodes explicitly instead of receiving made-up phases.
    """
    g = grip_signal(episode)
    closed = g > CLOSED_ABOVE
    if not closed.any():
        raise ValueError(f"{episode.episode_id}: left hand never closes (max g={g.max():.3f})")
    runs, start = [], None
    for i, value in enumerate(closed):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(g)))
    close_end, open_start = max(runs, key=lambda run: run[1] - run[0])
    close_start = close_end
    while close_start > 0 and g[close_start] > OPEN_BELOW:
        close_start -= 1
    open_end = min(open_start, len(g) - 1)
    while open_end < len(g) - 1 and g[open_end] > OPEN_BELOW:
        open_end += 1
    return Phases(close_start, close_end, open_start, open_end, len(g))


def count_grasps(episode: Episode) -> int:
    """Number of open-to-closing transitions of g(t)."""
    g = grip_signal(episode)
    return int(((g[1:] > OPEN_BELOW) & (g[:-1] <= OPEN_BELOW)).sum())
