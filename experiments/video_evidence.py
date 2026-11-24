"""Report figures from the 1st-dataset videos (evidence only; videos are never a model input).

The videos are not in the repository. Each mp4 has exactly as many frames as its CSV has rows,
so frame i shows the robot at CSV row i and log events (hand starts opening = open_start from
src.phases) can be shown directly.

    pip install opencv-python-headless        # only needed for this script
    python -m experiments.video_evidence --video-dir <folder with episode_000001.mp4 ...>

Outputs (reports/video_evidence/):
  alignment.csv             video frames vs CSV rows per episode
  failure_modes.jpg         success vs fail type A vs fail type B around the release
  cluster_v2_examples.jpg   episodes closest to each v2 cluster centre (src.cluster_v2)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import score_v2
from src.cluster_v2 import NAMES, MechanismClusterer
from src.data import load_dataset
from src.phases import segment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/video_evidence"
CROP = (40, 320, 150, 530)   # y0, y1, x0, x1 of the 640x360 frame: box and release area
OFFSETS = [(-15, "0.5 s before release"), (0, "hand starts opening"), (9, "+0.3 s"), (30, "+1.0 s"), (None, "end")]


def read_frames(cv2, path: Path, wanted: set[int]) -> tuple[dict[int, np.ndarray], int]:
    cap = cv2.VideoCapture(str(path))
    frames, i = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = frame
        i += 1
    cap.release()
    return frames, i


def strip(cv2, video: Path, episode, title: str) -> np.ndarray:
    p = segment(episode)
    idx = [p.n_frames - 1 if o is None else int(np.clip(p.open_start + o, 0, p.n_frames - 1)) for o, _ in OFFSETS]
    frames, _ = read_frames(cv2, video, set(idx))
    y0, y1, x0, x1 = CROP
    tiles = []
    for (_, label), i in zip(OFFSETS, idx):
        tile = cv2.resize(frames[i][y0:y1, x0:x1], (304, 224), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (303, 18), (0, 0, 0), -1)
        cv2.putText(tile, f"{label}  f{i}", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        tiles.append(tile)
    band = np.full((30, 304 * len(tiles), 3), 255, np.uint8)
    cv2.putText(band, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return np.vstack([band, np.hstack(tiles)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        import cv2
    except ImportError:
        raise SystemExit("This script needs OpenCV: pip install opencv-python-headless")

    episodes = {int(e.episode_id[-6:]): e for e in load_dataset(ROOT / "data")}
    video = {n: args.video_dir / f"episode_{n:06d}.mp4" for n in episodes}
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for n, e in sorted(episodes.items()):
        cap = cv2.VideoCapture(str(video[n]))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
        rows.append({"episode": n, "label": e.label, "csv_rows": len(e.frames), "video_frames": frames, "fps": fps})
    alignment = pd.DataFrame(rows)
    alignment.to_csv(OUT / "alignment.csv", index=False)
    print(f"videos aligned with CSV rows: {(alignment.csv_rows == alignment.video_frames).sum()}/{len(alignment)}")

    cases = [(34, "ep34 SUCCESS: lowered into the box, hand opens and leaves"),
             (12, "ep12 FAIL type A: released high above the rim / tilted (episodes 1-28)"),
             (25, "ep25 FAIL type B: stood on the floor, tipped while the hand opened (episodes 19-48)")]
    sheet = np.vstack([strip(cv2, video[n], episodes[n], t) for n, t in cases])
    cv2.imwrite(str(OUT / "failure_modes.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    model = score_v2.load()
    ids = sorted(episodes)
    F = pd.DataFrame([score_v2.score_inputs_v2(episodes[n]) for n in ids], index=ids)
    Z = model.group_z(F)
    clusterer = MechanismClusterer().fit(Z)
    cluster = pd.Series(clusterer.predict(Z), index=ids)
    blocks = []
    for c in range(3):
        members = cluster.index[cluster == c]
        dist = ((Z.loc[members] - clusterer.centres_.loc[c]) ** 2).sum(axis=1).sort_values()
        picks = []
        for label in (1, 0):   # closest success and closest fail, when the cluster has them
            same = [n for n in dist.index if episodes[n].label == label]
            if same:
                picks.append(same[0])
        rate = np.mean([episodes[n].label for n in members])
        for n in picks:
            title = f"C{c} {NAMES[c]} (n={len(members)}, success {rate:.0%}) - ep{n} {'success' if episodes[n].label else 'fail'}"
            blocks.append(strip(cv2, video[n], episodes[n], title))
    cv2.imwrite(str(OUT / "cluster_v2_examples.jpg"), np.vstack(blocks), [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"Saved figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
