# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING

from logzero import logger
from prometheus_client import Counter, Gauge, Histogram
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import io
matplotlib.use('agg')

from .to_scout import Label
from .utils import tailf

if TYPE_CHECKING:
    from .to_scout import ScoutQueue


HAWK_LABEL_POSITIVE = Counter(
    "hawk_label_positive",
    "Number of samples that were labeled as True Positive",
    labelnames=["mission", "labeler"],
)
HAWK_LABEL_NEGATIVE = Counter(
    "hawk_label_negative",
    "Number of samples that were labeled as False Positive",
    labelnames=["mission", "labeler"],
)
HAWK_LABEL_MSGSIZE = Counter(
    "hawk_label_msgsize",
    "Message size of (labeled) samples received from scouts",
    labelnames=["mission", "labeler"],
)
HAWK_LABEL_QUEUE_LENGTH = Gauge(
    "hawk_label_queue_length",
    "Number of samples waiting to be labeled",
    labelnames=["mission", "labeler"],
)
HAWK_LABEL_QUEUED_TIME = Histogram(
    "hawk_label_queued_time",
    "Time elapsed until a sample is labeled (seconds)",
    labelnames=["mission", "labeler"],
    buckets=(0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 25.0, 50.0, 75.0, 100.0),
)


@dataclass
class LabelerDiskQueue:
    mission_id: str
    scout_queue: ScoutQueue
    mission_dir: Path
    label_queue_size: int = 0

    token_semaphore: threading.BoundedSemaphore = field(init=False, repr=False)

    negatives: Counter = field(init=False)
    positives: Counter = field(init=False)
    totalsize: Counter = field(init=False)
    queue_length: Gauge = field(init=False)

    def __post_init__(self) -> None:
        if self.label_queue_size <= 0:
            self.label_queue_size = 9999

        # track number of unlabeled items being handled by the labeler
        self.token_semaphore = threading.BoundedSemaphore(self.label_queue_size)

        # track labeler statistics
        self.negatives = HAWK_LABEL_NEGATIVE.labels(
            mission=self.mission_id, labeler="disk"
        )
        self.positives = HAWK_LABEL_POSITIVE.labels(
            mission=self.mission_id, labeler="disk"
        )
        self.totalsize = HAWK_LABEL_MSGSIZE.labels(
            mission=self.mission_id, labeler="disk"
        )
        self.queue_length = HAWK_LABEL_QUEUE_LENGTH.labels(
            mission=self.mission_id, labeler="disk"
        )
        self.queued_time = HAWK_LABEL_QUEUED_TIME.labels(
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

        for index in count():
            # block until the labeler is ready to accept more.
            self.token_semaphore.acquire()

            # pull the next result from the priority queue
            result = self.scout_queue.get()

            logger.info(
                f"Labeling {result.object_id} {result.scout_index} {result.score}"
            )

            # write result image file to disk
            tile_jpeg = tile_dir.joinpath(f"{index:06}.jpeg")
            if result.object_id.endswith('.npy'): ## for radar missions with .npy files
                self.gen_heatmap(result.data, tile_jpeg)
            else: 
                tile_jpeg.write_bytes(result.data)
            logger.info(f"SAVED TILE {tile_jpeg}")

            # append metadata to unlabeled.jsonl file
            meta_json = result.to_json(index=index, queued_time=time.time())
            with unlabeled_jsonl.open("a") as f:
                f.write(f"{meta_json}\n")

            # update label_queue_length metric
            self.queue_length.inc()

            # logger.info(f"Meta: {count:06} {meta_json}")
            self.scout_queue.task_done()

    def labeler_to_home(self) -> None:
        labeled_jsonl = self.mission_dir / "labeled.jsonl"
        labeled_jsonl.touch()

        logger.info("Started reading labeled results from labeler")

        # read label results and forward to the scouts
        with labeled_jsonl.open() as fp:
            # Seek to the end of the labeled.jsonl file, assuming we've already
            # forwarded all labels, there really is no good way to track what
            # we've done because the semaphore we use doesn't survive a restart.
            fp.seek(0, 2)

            for label_json in tailf(fp):
                now = time.time()

                # update label_queue_length metric
                self.queue_length.dec()

                # release next unlabeled result to labeler
                self.token_semaphore.release()

                label = Label.from_json(label_json)
                self.scout_queue.put(label)

                # update stats
                if label.image_label in [None, "", "0", 0, 0.0]:
                    self.negatives.inc()
                else:
                    self.positives.inc()

                # not sure why we track this here and not where the unlabeled
                # tiles are received and queued for labeling.
                self.totalsize.inc(label.size)

                # track time it took to apply label
                if label.queued_time is not None:
                    queue_elapsed = now - label.queued_time
                    self.queued_time.observe(queue_elapsed)

    def gen_heatmap(self, data, tile_path):
        with io.BytesIO(data) as bytes_file:
            data = np.load(bytes_file,allow_pickle=True)
        logger.info(f"Array shape: {data.shape}")
        plt.imshow(data.sum(axis=2).transpose(), cmap='viridis', interpolation='nearest')
        plt.savefig(tile_path, bbox_inches='tight')
        plt.close('all')