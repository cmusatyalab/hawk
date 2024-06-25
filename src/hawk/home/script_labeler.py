# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, cast

import pandas as pd
from logzero import logger

from hawk.home.label_utils import MissionResults

if TYPE_CHECKING:
    from hawk.mission_config import MissionConfig

BBoxes = List[Tuple[float, float, float, float, float]]


@dataclass
class ScriptLabeler:
    mission_dir: Path
    class_list: list[str]
    label_time: float = 0.0
    detect: bool = False
    gt_path: Path | None = None
    positives = 0
    negatives = 0
    class_counter: list[int] = field(init=False)

    def __post_init__(self) -> None:
        if self.detect:
            assert self.gt_path is not None, "Ground Truth directory not specified"
            assert self.gt_path.exists(), "Ground Truth directory does not exist"

        self.labeling_func = self.classify_func if not self.detect else self.detect_func
        self.class_counter = [
            0 for i in range(len(self.class_list))
        ]  ## for multiclass support

    def classify_func(self, objectId: str) -> tuple[int, BBoxes]:
        return (
            int(objectId.split("/")[1]),
            [],
        )  ## returns the label regardless if 0, 1, 2 (handles multiclass)

    def detect_func(self, objectId: str) -> tuple[int, BBoxes]:
        assert self.gt_path is not None
        gt_name = Path(objectId).with_suffix(".txt").name
        gt_file = self.gt_path / gt_name
        if not gt_file.exists():
            return 0, []

        bounding_boxes = cast(
            BBoxes,
            [
                tuple(float(c) for c in line.split(" ", 5))
                for line in gt_file.read_text().splitlines()
            ],
        )
        classes = [int(box[0]) for box in bounding_boxes]
        labelClass = 1 if sum(classes) else 0
        return labelClass, bounding_boxes

    def run(self) -> None:
        self.mission_data = MissionResults(self.mission_dir, sync_labels=True)
        with suppress(KeyboardInterrupt):
            logger.debug("Waiting for data to label")
            for new_data in self.mission_data:
                new_data = new_data[new_data.imageLabel.isna()]
                logger.debug(f"Received {new_data.index.size} results to label")
                self.label_data(new_data)

    def label_data(self, data: pd.DataFrame) -> None:
        for index, objectId in data.objectId.items():
            imageLabel, boundingBoxes = self.labeling_func(objectId)

            new_label = pd.Series([imageLabel], index=[index], dtype=int)
            new_bboxes = pd.Series([boundingBoxes], index=[index])

            time.sleep(self.label_time)

            if imageLabel:
                self.positives += 1
                self.class_counter[imageLabel] += 1

            else:
                self.negatives += 1
                self.class_counter[0] += 1

            logger.info(
                f"Labeling {index:06} {imageLabel} {objectId}, "
                f"(Pos, Neg): ({self.positives}, {self.negatives})"
            )
            logger.info(f"By class: ({self.class_list}, {self.class_counter})")
            self.mission_data.save_new_labels(new_label, new_bboxes)

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

        dataset = config.get("dataset", {})
        class_list = dataset.get("class_list", ["positive", "negative"])
        logger.info(f"Class list: {class_list}")

        return cls(mission_dir, class_list, label_time, label_mode == "detect", gt_dir)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--label-time", type=float, default=1.0)
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--gt-path", type=Path)
    parser.add_argument("mission_directory", type=Path, nargs="?", default=".")
    args = parser.parse_args()

    ScriptLabeler(
        args.mission_directory, args.label_time, args.detect, args.gt_path
    ).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
