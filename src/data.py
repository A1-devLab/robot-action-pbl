"""Read labeled episodes without changing raw CSVs (Python standard library only)."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


JOINTS = tuple(
    f"{side}_{joint}"
    for side in ("l", "r")
    for joint in (
        "sh_p", "sh_r", "sh_y", "el_p", "wr_y", "wr_p", "wr_r",
        "hand_index", "hand_middle", "hand_pinky", "hand_ring",
        "hand_thumb_y", "hand_thumb_p",
    )
)
MOTION_COLUMNS = tuple(f"{kind}.{joint}" for kind in ("state", "action") for joint in JOINTS)
EXPECTED_COLUMNS = ("episode_index", "frame_index", "index", "task_index", "timestamp") + MOTION_COLUMNS
LABELS = {"fail": 0, "success": 1}


@dataclass
class Episode:
    """Metadata stays separate from the explicitly selected motion inputs."""

    episode_id: str
    label: int
    path: Path
    frames: list[dict[str, float]]
    summary: dict

    def motion_inputs(self) -> list[list[float]]:
        """Return 52 state/action values per frame, excluding IDs, paths and labels.

        These are raw motion values, not yet episode-level model features.
        """
        if self.summary["issues"]:
            raise ValueError(f"Invalid episode {self.episode_id}: {self.summary['issues']}")
        return [[row[column] for column in MOTION_COLUMNS] for row in self.frames]


def _read_episode(path: Path, label: int) -> Episode:
    issues: list[str] = []
    frames: list[dict[str, float]] = []
    missing = nonfinite = nonnumeric = malformed = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        columns = next(reader, [])
        schema_ok = len(columns) == len(set(columns)) and set(columns) == set(EXPECTED_COLUMNS)
        if not schema_ok:
            issues.append("column schema mismatch (missing, extra or duplicate columns)")
        for values in reader:
            if not values:  # Ignore physically blank lines, not empty data fields.
                continue
            malformed += len(values) != len(columns)
            row = {}
            for position, column in enumerate(columns):
                value = values[position].strip() if position < len(values) else ""
                if not value:
                    missing += 1
                    number = math.nan
                else:
                    try:
                        number = float(value)
                    except ValueError:
                        nonnumeric += 1
                        number = math.nan
                    else:
                        nonfinite += not math.isfinite(number)
                row[column] = number
            frames.append(row)
    if not frames:
        issues.append("empty episode")
    for count, name in ((missing, "missing values"), (nonfinite, "nonfinite values"),
                        (nonnumeric, "nonnumeric values"), (malformed, "malformed rows")):
        if count:
            issues.append(f"{name}: {count}")

    def series(column: str) -> list[float]:
        return [row.get(column, math.nan) for row in frames]

    times = series("timestamp")
    time_ok = bool(times) and all(math.isfinite(t) for t in times) and all(
        b > a for a, b in zip(times, times[1:])
    )
    if not time_ok:
        issues.append("timestamps must be finite and strictly increasing")
    indices = series("frame_index")
    frame_ok = bool(indices) and all(math.isfinite(i) and i.is_integer() for i in indices) and all(
        b == a + 1 for a, b in zip(indices, indices[1:])
    )
    if not frame_ok:
        issues.append("frame indices must be consecutive integers")
    match = re.fullmatch(r"episode_(\d+)", path.stem)
    id_ok = bool(match) and bool(frames) and all(
        i == int(match.group(1)) for i in series("episode_index")
    )
    if not id_ok:
        issues.append("filename and episode_index must identify the same episode")
    summary = {
        "episode_id": path.stem, "label": label, "class_name": path.parent.name,
        "path": str(path.resolve()), "n_frames": len(frames),
        "duration_sec": times[-1] - times[0] if time_ok else "",
        "n_columns": len(columns), "missing_count": missing,
        "nonfinite_count": nonfinite, "nonnumeric_count": nonnumeric,
        "malformed_row_count": malformed, "schema_ok": schema_ok,
        "time_order_ok": time_ok, "frame_sequence_ok": frame_ok,
        "episode_id_ok": id_ok, "issues": "; ".join(issues),
    }
    return Episode(path.stem, label, path.resolve(), frames, summary)


def load_dataset(data_dir: str | Path, *, strict: bool = True) -> list[Episode]:
    """Load direct CSV children of fail/ and success/ in deterministic order.

    strict=False retains invalid episodes for inspection; motion_inputs still
    rejects them. Unreadable files and missing/empty class folders always fail.
    """
    root = Path(data_dir).resolve()
    episodes = []
    for class_name, label in LABELS.items():
        folder = root / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
        if not files:
            raise ValueError(f"No CSV files in {folder}")
        for path in files:
            try:
                episodes.append(_read_episode(path, label))
            except (OSError, UnicodeError, csv.Error) as error:
                raise ValueError(f"Cannot read {path}: {error}") from error
    # Numeric keys also catch aliases such as episode_1 and episode_000001.
    keys = [int(m.group(1)) if (m := re.fullmatch(r"episode_(\d+)", e.episode_id))
            else e.episode_id for e in episodes]
    counts = Counter(keys)
    for episode, key in zip(episodes, keys):
        if counts[key] > 1:
            episode.summary["issues"] = "; ".join(filter(None, (
                episode.summary["issues"], "duplicate episode ID across files")))
    invalid = [e for e in episodes if e.summary["issues"]]
    if strict and invalid:
        raise ValueError("Dataset validation failed:\n" + "\n".join(
            f"{e.path}: {e.summary['issues']}" for e in invalid))
    return episodes


def write_summary(episodes: list[Episode], output: str | Path) -> Path:
    """Write an inspection table; this is not a model input table."""
    if not episodes:
        raise ValueError("Cannot summarize an empty dataset")
    target = Path(output).resolve()
    if any(target == e.path for e in episodes):
        raise ValueError("Summary output must not overwrite an input CSV")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episodes[0].summary))
        writer.writeheader()
        writer.writerows(e.summary for e in episodes)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--summary", type=Path, default=Path("reports/episode_summary.csv"))
    args = parser.parse_args()
    try:
        episodes = load_dataset(args.data_dir, strict=False)
        output = write_summary(episodes, args.summary)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    counts = Counter(e.label for e in episodes)
    invalid = sum(bool(e.summary["issues"]) for e in episodes)
    print(f"Episodes: {len(episodes)} | success: {counts[1]} | fail: {counts[0]}")
    print(f"Frames: {sum(len(e.frames) for e in episodes)} | invalid episodes: {invalid}")
    print(f"Summary: {output}")
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
