# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from logzero import logger

from hawk.classes import NEGATIVE_CLASS, ClassCounter, ClassList
from hawk.home.label_utils import LabelSample, MissionData

if TYPE_CHECKING:
    from hawk.mission_config import MissionConfig


@dataclass
class ScriptLabeler:
    mission_dir: Path
    class_list: ClassList
    label_time: float = 0.0
    avoid_duplicates: bool = True
    positives: int = 0
    negatives: int = 0

    def __post_init__(self) -> None:
        self.class_counter = ClassCounter(self.class_list)

    def run(self) -> None:
        self.mission_data = MissionData(self.mission_dir)
        with suppress(KeyboardInterrupt):
            logger.debug("Waiting for data to label")
            for result in self.mission_data.read_unlabeled(
                exclude_labeled=self.avoid_duplicates, tail=True
            ):
                logger.debug("Received new results to label")
                self.label_data(result)
                time.sleep(self.label_time)

    def label_data(self, result: LabelSample) -> None:
        # unlabeled inference results from the scouts must always have an objectId
        assert result.objectId is not None

        result.detections = result.groundtruth

        # if there are multiple detections for the same class we count all of them
        counts = result.class_counts()
        self.class_counter.update(counts)

        del counts[NEGATIVE_CLASS]
        if counts:
            self.positives += 1
        else:
            self.negatives += 1

        self.class_list.extend(counts.keys())

        logger.info(
            f"Labeling {result.index:06} {counts} {result.objectId}, "
            f"(Pos, Neg): ({self.class_counter.positives},"
            f" {self.class_counter.negatives})"
        )
        logger.info(f"By class: {self.class_counter!r}")

        self.mission_data.save_labeled([result])

    @classmethod
    def from_mission_config(
        cls,
        config: MissionConfig,
        mission_dir: Path,
    ) -> ScriptLabeler:
        label_time = 2.0

        selector = config.get("selector", {})
        if selector.get("type") == "token":
            label_time = float(selector["token"].get("label_time", 1.0))

        class_names = config.get("dataset", {}).get("class_list", ["positive"])
        class_list = ClassList(class_names)

        return cls(
            mission_dir,
            class_list,
            label_time,
            avoid_duplicates=False,
        )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--label-time", type=float, default=1.0)
    parser.add_argument("--label-class", action="append")
    parser.add_argument("mission_directory", type=Path, nargs="?", default=".")
    args = parser.parse_args()

    class_names = args.label_class or ["positive"]
    class_list = ClassList(class_names)

    ScriptLabeler(
        args.mission_directory,
        class_list,
        args.label_time,
        avoid_duplicates=True,
    ).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
