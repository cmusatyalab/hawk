# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import sys
import time
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from logzero import logger

from hawk.home.label_utils import BoundingBox, LabelSample, MissionResults

if TYPE_CHECKING:
    from hawk.mission_config import MissionConfig


@dataclass
class ScriptLabeler:
    mission_dir: Path
    class_list: list[str]
    label_time: float = 0.0
    detect: bool = False
    gt_path: Path | None = None
    positives = 0
    negatives = 0
    class_counter: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        if self.detect:
            assert self.gt_path is not None, "Ground Truth directory not specified"
            assert self.gt_path.exists(), "Ground Truth directory does not exist"

        self.labeling_func = self.classify_func if not self.detect else self.detect_func

    def classify_func(self, objectId: str) -> list[BoundingBox]:
        if objectId.startswith("/0/"):
            return []
        class_index = int(objectId.split("/", 2)[1])
        return [BoundingBox(label=class_index)]

    def detect_func(self, objectId: str) -> list[BoundingBox]:
        assert self.gt_path is not None
        gt_name = Path(objectId).with_suffix(".txt").name
        gt_file = self.gt_path / gt_name
        if not gt_file.exists():
            return []

        return [
            BoundingBox.from_yolo(line) for line in gt_file.read_text().splitlines()
        ]

    def run(self) -> None:
        self.mission_data = MissionResults(self.mission_dir)
        with suppress(KeyboardInterrupt):
            logger.debug("Waiting for data to label")
            for result in self.mission_data.read_unlabeled(tail=True):
                logger.debug("Received new results to label")
                self.label_data(result)
                time.sleep(self.label_time)

    def label_data(self, result: LabelSample) -> None:
        result.labels = self.labeling_func(result.objectId)

        labels = list({bbox.label for bbox in result.labels})
        if labels:
            self.positives += 1
            for class_index in labels:
                class_name = self.class_list[class_index]
                self.class_counter[class_name] += 1
        else:
            self.negatives += 1
            self.class_counter["negative"] += 1

        logger.info(
            f"Labeling {result.index:06} {labels} {result.objectId}, "
            f"(Pos, Neg): ({self.positives}, {self.negatives})"
        )
        logger.info(f"By class: {list(self.class_counter.items())}")

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

        class_list = config.get("dataset", {}).get(
            "class_list", ["negative", "positive"]
        )

        return cls(mission_dir, class_list, label_time, label_mode == "detect", gt_dir)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--label-time", type=float, default=1.0)
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--label-class", action="append")
    parser.add_argument("--gt-path", type=Path)
    parser.add_argument("mission_directory", type=Path, nargs="?", default=".")
    args = parser.parse_args()

    class_list = ["negative"] + (args.label_class or ["positive"])

    ScriptLabeler(
        args.mission_directory, class_list, args.label_time, args.detect, args.gt_path
    ).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
