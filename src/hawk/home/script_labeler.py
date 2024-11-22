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

from hawk.classes import NEGATIVE_CLASS, ClassCounter, ClassList, ClassName
from hawk.home.label_utils import Detection, LabelSample, MissionResults

if TYPE_CHECKING:
    from hawk.mission_config import MissionConfig


@dataclass
class ScriptLabeler:
    mission_dir: Path
    class_list: ClassList
    label_time: float = 0.0
    detect: bool = False
    gt_path: Path | None = None
    avoid_duplicates: bool = True
    positives: int = 0
    negatives: int = 0

    def __post_init__(self) -> None:
        if self.detect:
            assert self.gt_path is not None, "Ground Truth directory not specified"
            assert self.gt_path.exists(), "Ground Truth directory does not exist"

        self.class_counter = ClassCounter(self.class_list)
        self.labeling_func = self.classify_func if not self.detect else self.detect_func

    def classify_func(self, objectId: str) -> list[Detection]:
        if objectId.startswith("/negative/"):
            return []
        class_value = objectId.split("/", 2)[1]
        class_name = ClassName(sys.intern(class_value))
        self.class_list.add(class_name)
        return [Detection(scores={class_name: 1.0})]

    def detect_func(self, objectId: str) -> list[Detection]:
        assert self.gt_path is not None
        gt_name = Path(objectId).with_suffix(".txt").name
        gt_file = self.gt_path / gt_name
        if not gt_file.exists():
            return []

        return list(
            Detection.merge_detections(
                Detection.from_yolo(line, self.class_list)
                for line in gt_file.read_text().splitlines()
            )
        )

    def run(self) -> None:
        self.mission_data = MissionResults(self.mission_dir)
        with suppress(KeyboardInterrupt):
            logger.debug("Waiting for data to label")
            for result in self.mission_data.read_unlabeled(
                exclude_labeled=self.avoid_duplicates, tail=True
            ):
                logger.debug("Received new results to label")
                self.label_data(result)
                time.sleep(self.label_time)

    def label_data(self, result: LabelSample) -> None:
        result.detections = self.labeling_func(result.objectId)

        # if there are multiple detections for the same class we count all of them
        labels = list(
            cls for detection in result.detections for cls in detection.scores
        )
        if labels:
            self.positives += 1
            self.class_counter.update(labels)
        else:
            self.negatives += 1
            self.class_counter.update([NEGATIVE_CLASS])

        logger.info(
            f"Labeling {result.index:06} {labels} {result.objectId}, "
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
        label_time = 0.0

        selector = config.get("selector", {})
        if selector.get("type") == "token":
            label_time = float(selector["token"].get("label_time", 1.0))

        trainer = (config["train_strategy"]["type"]).lower()

        label_mode = "classify"

        if trainer == "dnn_classifier_radar":
            if config["train_strategy"]["args"]["pick_patches"] == "True":
                label_mode = "detect"
        elif trainer == "yolo":
            label_mode = "detect"

        gt_dir = Path(config["home-params"].get("label_dir", ""))
        # logger.info(f"GT DIR: {gt_dir}, {type(gt_dir)}")

        class_names = config.get("dataset", {}).get("class_list", ["positive"])
        class_list = ClassList(class_names)

        return cls(
            mission_dir,
            class_list,
            label_time,
            label_mode == "detect",
            gt_dir,
            avoid_duplicates=False,
        )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--label-time", type=float, default=1.0)
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--label-class", action="append")
    parser.add_argument("--gt-path", type=Path)
    parser.add_argument("mission_directory", type=Path, nargs="?", default=".")
    args = parser.parse_args()

    class_names = args.label_class or ["positive"]
    class_list = ClassList(class_names)

    ScriptLabeler(
        args.mission_directory,
        class_list,
        args.label_time,
        args.detect,
        args.gt_path,
        avoid_duplicates=True,
    ).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
