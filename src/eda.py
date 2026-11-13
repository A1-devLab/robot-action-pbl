"""Plot all motion channels by class on normalized episode time."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import JOINTS, load_dataset


def plot_episodes(episodes, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0, 1, 101)
    for kind in ("state", "action"):
        fig, axes = plt.subplots(7, 4, figsize=(17, 22), sharex=True)
        for ax, joint in zip(axes.flat, JOINTS):
            for label, name, color in ((0, "fail", "#d55e00"), (1, "success", "#0072b2")):
                traces = []
                for episode in episodes:
                    if episode.label != label:
                        continue
                    time = np.array([row["timestamp"] for row in episode.frames])
                    time = (time - time[0]) / (time[-1] - time[0]) if len(time) > 1 else np.array([0.])
                    trace = np.interp(grid, time, [row[f"{kind}.{joint}"] for row in episode.frames])
                    traces.append(trace)
                    ax.plot(grid, trace, color=color, alpha=.12, linewidth=.7)
                ax.plot(grid, np.mean(traces, axis=0), label=f"{name} (n={len(traces)})", color=color, linewidth=2)
            ax.set_title(joint)
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(JOINTS):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"{kind}: thin = episode, thick = class mean\nNormalized time aligns duration, not task phases", fontsize=15)
        fig.supxlabel("Normalized episode time (0 = start, 1 = end)")
        fig.supylabel("Joint values (source units; verify units before interpretation)")
        fig.tight_layout(rect=(.02, .02, 1, .96))
        fig.savefig(output / f"{kind}_by_class.png", dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eda"))
    args = parser.parse_args()
    plot_episodes(load_dataset(args.data_dir), args.output_dir)
    print(f"Saved state/action comparison plots to {args.output_dir}")


if __name__ == "__main__":
    main()
