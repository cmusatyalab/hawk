# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from logzero import logger
from prometheus_client import Counter, Gauge, Histogram

from .label_utils import ClassMap, index_jsonl, read_jsonl
from .stats import (
    HAWK_LABELED_CLASSES,
    HAWK_LABELED_OBJECTS,
    HAWK_LABELER_QUEUED_LENGTH,
    HAWK_LABELER_QUEUED_TIME,
)

matplotlib.use("agg")

if TYPE_CHECKING:
    from .to_scout import ScoutQueue


@dataclass
class LabelerDiskQueue:
    mission_id: str
    scout_queue: ScoutQueue
    mission_dir: Path
    class_map: ClassMap
    label_queue_size: int = 0

    token_semaphore: threading.BoundedSemaphore = field(init=False, repr=False)

    labeled_objects: Histogram = field(init=False)
    labeled_classes: Counter = field(init=False)
    queue_length: Gauge = field(init=False)
    queued_time: Histogram = field(init=False)

    def __post_init__(self) -> None:
        if self.label_queue_size <= 0:
            self.label_queue_size = 9999

        # track number of unlabeled items being handled by the labeler
        self.token_semaphore = threading.BoundedSemaphore(self.label_queue_size)

        # track labeler statistics
        self.labeled_objects = HAWK_LABELED_OBJECTS.labels(
            mission=self.mission_id, labeler="disk"
        )

        self.labeled_classes = HAWK_LABELED_CLASSES
        # Hint to prometheus_client which class names we may use later
        for class_name in self.class_map.classes:
            HAWK_LABELED_CLASSES.labels(
                mission=self.mission_id, labeler="disk", class_name=class_name
            )

        self.queue_length = HAWK_LABELER_QUEUED_LENGTH.labels(
            mission=self.mission_id, labeler="disk"
        )
        self.queued_time = HAWK_LABELER_QUEUED_TIME.labels(
            mission=self.mission_id, labeler="disk"
        )

    def start(self) -> LabelerDiskQueue:
        # start the threads that interact with the labeler
        threading.Thread(target=self.labeler_to_home, daemon=True).start()
        threading.Thread(target=self.home_to_labeler, daemon=True).start()
        return self

    def home_to_labeler(self) -> None:
        unlabeled_jsonl = self.mission_dir / "unlabeled.jsonl"
        tile_dir = self.mission_dir / "images"
        tile_dir.mkdir(exist_ok=True)
        fv_dir = self.mission_dir / "feature_vectors"
        fv_dir.mkdir(exist_ok=True)

        # check how many unlabeled samples we have already processed
        # (we may have restarted the hawk_label_broker process)
        # this can be much faster if we don't actually try to index/json decode
        # but the assumption is that most of the time the file will be empty
        _, lines = index_jsonl(unlabeled_jsonl)

        for index in count(lines + 1):
            # block until the labeler is ready to accept more.
            self.token_semaphore.acquire()

            # pull the next result from the priority queue
            result = self.scout_queue.get()

            logger.info(
                f"Labeling {result.objectId} {result.scoutIndex} {result.score}"
            )

            # write result image file to disk
            tile_jpeg = tile_dir.joinpath(f"{index:06}.jpeg")
            feature_vector_path = fv_dir.joinpath(f"{index:06}.pt")
            if result.objectId.endswith(".npy"):  # for radar missions with .npy files
                self.gen_heatmap(result.data, tile_jpeg)
            else:
                tile_jpeg.write_bytes(result.data)
            logger.info(f"SAVED TILE {tile_jpeg}")
            if result.feature_vector:
                torch.save(result.feature_vector, feature_vector_path)

            # update queued time so we can track labeling delay.
            result.queued = time.time()

            # update label_queue_length metric
            self.queue_length.inc()

            # append metadata to unlabeled.jsonl file
            with unlabeled_jsonl.open("a") as fp:
                result.to_jsonl(fp, self.class_map)

            # logger.info(f"Meta: {count:06} {meta_json}")
            self.scout_queue.task_done()

    def labeler_to_home(self) -> None:
        labeled_jsonl = self.mission_dir / "labeled.jsonl"
        labeled_jsonl.touch()

        logger.info("Started reading labeled results from labeler")

        # skip previously labeled results
        labeled, _ = index_jsonl(labeled_jsonl)

        # read labeled results and forward to the scouts
        for result in read_jsonl(labeled_jsonl, exclude=labeled, tail=True):
            now = time.time()

            # update label_queue_length metric
            self.queue_length.dec()

            # release next unlabeled result to labeler
            self.token_semaphore.release()

            self.scout_queue.put(result)

            # update stats
            detections = len(result.detections)
            self.labeled_objects.observe(detections)

            for detection in result.detections:
                for cls in detection.cls_scores:
                    self.labeled_classes.labels(
                        mission=self.mission_id,
                        labeler="disk",
                        class_name=cls,
                    ).inc()

            # track time it took to apply label
            if result.queued is not None:
                queue_elapsed = now - result.queued
                self.queued_time.observe(queue_elapsed)

    def gen_heatmap(self, data_: bytes, tile_path: Path) -> None:
        with io.BytesIO(data_) as bytes_file:
            data = np.load(bytes_file, allow_pickle=True)
        plt.imshow(
            data.sum(axis=2).transpose(), cmap="viridis", interpolation="nearest"
        )
        plt.xticks([0, 16, 32, 48, 63], [-13, -6.5, 0, 6.5, 13], fontsize=8)
        plt.yticks([0, 64, 128, 192, 255], [50, 37.5, 25, 12.5, 0])
        plt.xlabel("velocity (m/s)")
        plt.ylabel("range (m)")
        # plt.title("RD Map")
        plt.savefig(tile_path, bbox_inches="tight")
        plt.close("all")
