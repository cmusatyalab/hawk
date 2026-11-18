#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only
"""Per-class results of finished Hawk missions.

    python sar_results.py MISSION_DIR [MISSION_DIR ...] [--csv OUT.csv]

MISSION_DIR is a folder that hawk_home wrote (it holds labeled.jsonl and
logs/). For each mission this prints how many positives of every class were
labeled against how many were in the part of the stream the scouts scored,
which class the labels went to per model version, and how the finds built up
over the mission. With more than one mission it also prints a comparison,
including the worst per-class recall, the number a class-aware selector should
raise.

--csv writes one row per labeled tile (mission, order, time since the first
label, model version, class, and the running count of that class), for
plotting finds over time.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

NEGATIVE = "negative"


class Mission:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = path.name
        lines = (path / "labeled.jsonl").read_text().splitlines()
        self.labels = [json.loads(line) for line in lines if line]
        self.labels.sort(key=lambda label: label["queued"])
        self.stats = self._final_stats()
        self.config = self._config()

        # the classes this mission searched for, in config order
        self.classes = list(self.config.get("dataset", {}).get("class_list", []))
        if not self.classes:
            self.classes = sorted(self.stats.get("count_by_class", {}))
        self.in_stream = {
            c: int(self.stats.get(f"positive_in_stream_{c}", 0)) for c in self.classes
        }
        self.found = Counter(map(label_class, self.labels))

    def _final_stats(self) -> dict:
        logs = self.path / "logs"
        for name in ("mission-stats.json", "final-stats.json"):
            stats_file = logs / name
            if stats_file.exists():
                return json.loads(stats_file.read_text())
        stats_files = sorted(logs.glob("stats-*.json"))
        if not stats_files:
            sys.exit(f"{self.path}: no stats in logs/")
        return json.loads(stats_files[-1].read_text())

    def _config(self) -> dict:
        config_file = self.path / "logs" / "hawk.yml"
        if not config_file.exists():
            return {}
        return yaml.safe_load(config_file.read_text()) or {}

    @property
    def selector(self) -> str:
        selector = self.config.get("selector", {})
        kind = selector.get("type", "?")
        options = selector.get(kind, {}) if isinstance(selector.get(kind), dict) else {}
        if "scheduler" in options:
            kind += f"/{options['scheduler']}"
        if options.get("warm_start"):
            kind += f"+warm{options['warm_start']}"
        k, batch = options.get("k"), options.get("batchSize")
        return f"{kind} k={k}/{batch}" if k and batch else kind

    @property
    def people(self) -> int:
        return sum(self.found[c] for c in self.classes)

    @property
    def people_in_stream(self) -> int:
        return sum(self.in_stream.values())

    def recall(self, class_name: str) -> float:
        total = self.in_stream[class_name]
        return self.found[class_name] / total if total else 0.0

    @property
    def worst(self) -> tuple[str, float]:
        worst = min(self.classes, key=self.recall)
        return worst, self.recall(worst)


def label_class(label: dict) -> str:
    groundtruth = label.get("groundtruth") or []
    return groundtruth[0]["class_name"] if groundtruth else NEGATIVE


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def report(m: Mission) -> None:
    scored = int(m.stats.get("processedObjects", 0))
    total = int(m.stats.get("totalObjects", 0))
    labels = len(m.labels)
    print(f"== {m.name}  ({m.selector})")
    print(
        f"scored {scored:,} of {total:,} tiles, {labels} labels, "
        f"{m.people} positives, {m.found[NEGATIVE]} negatives, "
        f"precision {pct(m.people / labels if labels else 0)}, "
        f"recall {pct(m.people / m.people_in_stream if m.people_in_stream else 0)}, "
        f"last model v{int(m.stats.get('version', 0))}",
    )
    print()
    header = ("class", "found", "in stream", "recall", "of finds", "of stream", "ratio")
    print("{:10s}{:>7s}{:>11s}{:>9s}{:>10s}{:>11s}{:>7s}".format(*header))
    for c in sorted(m.classes, key=lambda c: -m.in_stream[c]):
        share_found = m.found[c] / m.people if m.people else 0
        share_stream = m.in_stream[c] / m.people_in_stream if m.people_in_stream else 0
        ratio = share_found / share_stream if share_stream else 0
        print(
            f"{c:10s}{m.found[c]:7d}{m.in_stream[c]:11d}{pct(m.recall(c)):>9s}"
            f"{pct(share_found):>10s}{pct(share_stream):>11s}{ratio:7.2f}",
        )
    worst, worst_recall = m.worst
    print(f"worst class: {worst} at {pct(worst_recall)}")

    print("\nlabels per model version")
    by_version: dict[int, Counter[str]] = defaultdict(Counter)
    for label in m.labels:
        by_version[label["model_version"]][label_class(label)] += 1
    for version in sorted(by_version):
        counts = by_version[version]
        cells = "  ".join(f"{c} {counts[c]}" for c in [*m.classes, NEGATIVE])
        print(f" v{version:<3d}{sum(counts.values()):5d}  {cells}")

    print("\nfinds by quarter of the labels")
    for quarter in (1, 2, 3, 4):
        counts = Counter(map(label_class, m.labels[: labels * quarter // 4]))
        cells = "  ".join(f"{c} {counts[c]}" for c in m.classes)
        print(f" {25 * quarter:3d}%  {cells}")
    print()


def compare(missions: list[Mission]) -> None:
    classes = missions[0].classes
    print("== comparison (recall per class)")
    header = f"{'mission':34s}{'selector':28s}{'labels':>7s}{'found':>7s}"
    header += "".join(f"{c[:8]:>9s}" for c in classes) + f"{'worst':>9s}"
    print(header)
    for m in missions:
        row = f"{m.name[:33]:34s}{m.selector[:27]:28s}{len(m.labels):7d}{m.people:7d}"
        row += "".join(f"{pct(m.recall(c)):>9s}" for c in classes)
        print(row + f"{pct(m.worst[1]):>9s}")


def write_csv(missions: list[Mission], out: Path) -> None:
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["mission", "order", "seconds", "model_version", "class", "class_count"]
        )
        for m in missions:
            start = m.labels[0]["queued"] if m.labels else 0
            running: Counter[str] = Counter()
            for order, label in enumerate(m.labels, 1):
                class_name = label_class(label)
                running[class_name] += 1
                writer.writerow(
                    [
                        m.name,
                        order,
                        f"{label['queued'] - start:.1f}",
                        label["model_version"],
                        class_name,
                        running[class_name],
                    ],
                )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("missions", nargs="+", type=Path, help="mission folders")
    ap.add_argument("--csv", type=Path, help="write one row per label to this file")
    args = ap.parse_args()

    missions = [Mission(path) for path in args.missions]
    for m in missions:
        report(m)
    if len(missions) > 1:
        compare(missions)
    if args.csv:
        write_csv(missions, args.csv)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
