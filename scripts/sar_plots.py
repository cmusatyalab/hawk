#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only
"""Plots comparing selectors over finished Hawk missions.

    python sar_plots.py MISSION_DIR [MISSION_DIR ...] --out PLOT_DIR

Missions with the same selector (repeats with a reshuffled stream) are grouped;
the plots show their mean, with the range across repeats. Writes:

    recall_by_class.png   recall of every class, one bar per selector
    finds_over_labels.png running count of finds of every class over the labels
    recall_by_class.csv   the numbers behind the first plot

Needs sar_results.py next to this script.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sar_results import Mission, label_class  # noqa: E402

# categorical slots in fixed order; each selector owns a slot, so it keeps its
# color whichever runs a plot includes
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SELECTORS = ["topk", "token", "class_aware/round_robin", "class_aware/deficit"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


def selector_name(m: Mission) -> str:
    """Selector label without k and batch size, e.g. class_aware/deficit."""
    return m.selector.split(" ")[0]


def color(name: str) -> str:
    if name not in SELECTORS:
        SELECTORS.append(name)
    return COLORS[SELECTORS.index(name) % len(COLORS)]


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def recall_plot(
    groups: dict[str, list[Mission]], classes: list[str], out: Path
) -> None:
    names = list(groups)
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    style(ax)
    rows = []
    for i, name in enumerate(names):
        recalls = [[100 * m.recall(c) for m in groups[name]] for c in classes]
        means = [sum(r) / len(r) for r in recalls]
        lows = [mean - min(r) for mean, r in zip(means, recalls)]
        highs = [max(r) - mean for mean, r in zip(means, recalls)]
        xs = [x + (i - (len(names) - 1) / 2) * width for x in range(len(classes))]
        ax.bar(
            xs,
            means,
            width * 0.9,  # a gap between neighbouring bars
            color=color(name),
            label=f"{name} ({len(groups[name])} runs)",
            yerr=[lows, highs] if len(groups[name]) > 1 else None,
            error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 2},
        )
        runs = len(groups[name])
        rows += [[name, c, f"{v:.1f}", runs] for c, v in zip(classes, means)]
    ax.set_xticks(range(len(classes)), classes, color=INK)
    ax.set_ylabel("recall (%)", color=INK_SECONDARY)
    ax.set_title(
        "Recall per class (mean, bars span the repeats)",
        color=INK,
        loc="left",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "recall_by_class.png", dpi=160)
    plt.close(fig)
    with open(out / "recall_by_class.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["selector", "class", "mean_recall_pct", "runs"])
        writer.writerows(rows)


def finds_plot(
    groups: dict[str, list[Mission]], classes: list[str], ref: Mission, out: Path
) -> None:
    fig, axes = plt.subplots(
        1, len(classes), figsize=(3 * len(classes), 3.2), facecolor=SURFACE
    )
    for ax, class_name in zip(axes, classes):
        style(ax)
        for name, missions in groups.items():
            # mean running count over the repeats, up to the shortest run
            length = min(len(m.labels) for m in missions)
            totals = [0.0] * length
            for m in missions:
                count = 0
                for n, label in enumerate(m.labels[:length]):
                    count += label_class(label) == class_name
                    totals[n] += count
            ys = [t / len(missions) for t in totals]
            ax.plot(range(1, length + 1), ys, color=color(name), linewidth=2)
            ax.annotate(  # direct label at the line end
                f"{ys[-1]:.0f}",
                (length, ys[-1]),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=INK_SECONDARY,
            )
        in_stream = ref.in_stream[class_name]
        ax.set_title(f"{class_name} ({in_stream} in stream)", color=INK, fontsize=10)
        ax.set_xlabel("labels", color=INK_SECONDARY, fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel("found", color=INK_SECONDARY)
    handles = [plt.Line2D([], [], color=color(name), linewidth=2) for name in groups]
    fig.legend(
        handles,
        list(groups),
        loc="upper center",
        ncol=len(groups),
        frameon=False,
        fontsize=9,
        labelcolor=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "finds_over_labels.png", dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("missions", nargs="+", type=Path, help="mission folders")
    ap.add_argument("--out", type=Path, required=True, help="folder for the plots")
    args = ap.parse_args()

    groups: dict[str, list[Mission]] = defaultdict(list)
    for path in args.missions:
        m = Mission(path)
        groups[selector_name(m)].append(m)
    for name in groups:
        color(name)  # unknown selectors take the next free slots
    groups = dict(sorted(groups.items(), key=lambda g: SELECTORS.index(g[0])))
    # class counts from the run that scored the most of the stream; rarest
    # class last, so it sits at the right of every plot
    ref = max(
        (m for ms in groups.values() for m in ms), key=lambda m: m.people_in_stream
    )
    classes = sorted(ref.classes, key=lambda c: -ref.in_stream[c])

    args.out.mkdir(parents=True, exist_ok=True)
    recall_plot(groups, classes, args.out)
    finds_plot(groups, classes, ref, args.out)
    for name, missions in groups.items():
        worst = [m.worst[1] for m in missions]
        print(
            f"{name}: {len(missions)} runs, worst-class recall "
            f"{100 * sum(worst) / len(worst):.1f}% on average"
        )
    print(f"wrote plots to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
