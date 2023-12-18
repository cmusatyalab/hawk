# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import json
import queue
import time
from multiprocessing.synchronize import Event
from pathlib import Path

from logzero import logger

from ..mission_config import MissionConfig
from .typing import Labeler, LabelQueueType, MetaQueueType, StatsQueueType


class ScriptLabeler(Labeler):
    def __init__(
        self,
        label_dir: Path,
        configuration: MissionConfig,
        gt_path: str = "",
        label_mode: str = "classify",
    ) -> None:
        self._label_dir = label_dir
        self._gt_path = Path(gt_path)
        self._token = False
        self._label_time = 0.0
        # Token selector code to modify labeling process.
        self.configuration = configuration
        selector_field = self.configuration["selector"]
        if selector_field["type"] == "token":
            self._token = True
            init_samples = selector_field["token"]["initial_samples"]
            num_scouts = len(self.configuration["scouts"])
            self.total_init_samples = int(init_samples * int(num_scouts))
            self._label_time = float(selector_field["token"]["label_time"])
        ##########

        if label_mode == "classify":
            self.labeling_func = self.classify
        elif label_mode == "detect":
            self.labeling_func = self.detect
        else:
            raise NotImplementedError(f"Labeling mode not known {label_mode}")

    def start_labeling(
        self,
        input_q: MetaQueueType,
        result_q: LabelQueueType,
        stats_q: StatsQueueType,
        stop_event: Event,
    ) -> None:
        self.input_q = input_q
        self.result_q = result_q
        self.stop_event = stop_event
        self.stats_q = stats_q
        self.positives = 0
        self.negatives = 0
        self.bytes = 0
        self.received_samples = 0

        try:
            self.labeling_func()
        except KeyboardInterrupt as e:
            raise e

    def classify(self) -> None:
        # Object ID contains label
        # if /1/ in Id then label = 1 else 0
        try:
            while not self.stop_event.is_set():
                try:
                    # get the meta path for the next sample to label
                    meta_path = Path(self.input_q.get())
                    self.received_samples += 1
                except queue.Empty:
                    continue

                data_name = meta_path.name
                logger.info(data_name)

                label_path = self._label_dir / f"{data_name}"

                data = {}
                # get the data from the meta_data file
                with open(meta_path) as f:
                    data = json.load(f)

                image_label = "1" if "/1/" in data["objectId"] else "0"

                if image_label == "1":
                    self.positives += 1
                else:
                    self.negatives += 1
                time.sleep(self._label_time)

                self.bytes += data["size"]

                label = {
                    "objectId": data["objectId"],
                    "scoutIndex": data["scoutIndex"],
                    "imageLabel": image_label,
                    "boundingBoxes": [],
                }

                with open(label_path, "w") as f:
                    json.dump(label, f)

                self.result_q.put(str(label_path))
                logger.info(
                    "({}, {}) Labeled {}".format(
                        self.positives, self.negatives, data["objectId"]
                    )
                )
                self.stats_q.put((self.positives, self.negatives, self.bytes))
        except (OSError, KeyboardInterrupt) as e:
            logger.error(e)

    def detect(self) -> None:
        assert self._gt_path.exists(), "GT Dir does not exist"
        # Takes labels from file: _gt_path/<basename>.txt
        try:
            while not self.stop_event.is_set():
                try:
                    meta_path = Path(self.input_q.get())
                except queue.Empty:
                    continue

                data_name = meta_path.name

                logger.info(data_name)

                label_path = self._label_dir / f"{data_name}"

                data = {}
                with open(meta_path) as f:
                    data = json.load(f)

                basename = Path(data["objectId"]).name
                label_file = self._gt_path / (basename.split(".")[0] + ".txt")

                bounding_boxes = []
                if label_file.exists():
                    bounding_boxes = open(label_file).read().splitlines()

                if len(bounding_boxes):
                    image_label = "1"
                    self.positives += 1
                else:
                    image_label = "0"
                    self.negatives += 1

                self.bytes += data["size"]

                label = {
                    "objectId": data["objectId"],
                    "scoutIndex": data["scoutIndex"],
                    "imageLabel": image_label,
                    "boundingBoxes": bounding_boxes,
                }

                with open(label_path, "w") as f:
                    json.dump(label, f)

                self.result_q.put(str(label_path))
                logger.info(
                    "({}, {}) Labeled {}".format(
                        self.positives, self.negatives, data["objectId"]
                    )
                )
                self.stats_q.put((self.positives, self.negatives, self.bytes))
        except (OSError, KeyboardInterrupt) as e:
            logger.error(e)
