# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from logzero import logger
from prometheus_client import Counter, Gauge, Histogram

from ..classes import ClassList
from .label_utils import index_jsonl, read_jsonl
from .stats import (
    HAWK_LABELED_CLASSES,
    HAWK_LABELED_OBJECTS,
    HAWK_LABELER_QUEUED_LENGTH,
    HAWK_LABELER_QUEUED_TIME,
)

if TYPE_CHECKING:
    from .to_scout import ScoutQueue


@dataclass
class LabelerDiskQueue:
    mission_id: str
    scout_queue: ScoutQueue
    mission_dir: Path
    class_list: ClassList
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
        for class_name in self.class_list:
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

        while True:
            # block until the labeler is ready to accept more.
            self.token_semaphore.acquire()

            # pull the next result from the priority queue
            result = self.scout_queue.get()

            logger.info(
                f"Labeling {result.objectId} {result.scoutIndex} {result.score}"
            )

            # update queued time so we can track labeling delay.
            result.queued = time.time()

            # update label_queue_length metric
            self.queue_length.inc()

            # append metadata to unlabeled.jsonl file
            with unlabeled_jsonl.open("a") as fp:
                result.to_jsonl(fp, self.class_list)

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

            # This unassuming line is the main workhorse where we queue the new
            # label to be sent back to the scout where the sample originated.
            # The story continues in to_scout.HomeToScout...
            self.scout_queue.put(result)

            # The rest of this function is just updating stats
            detections = len(result.detections)
            self.labeled_objects.observe(detections)

            for detection in result.detections:
                for class_name in detection.scores:
                    self.labeled_classes.labels(
                        mission=self.mission_id,
                        labeler="disk",
                        class_name=class_name,
                    ).inc()

            # track time it took to apply label
            if result.queued is not None:
                queue_elapsed = now - result.queued
                self.queued_time.observe(queue_elapsed)
