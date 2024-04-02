# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING

from logzero import logger

from .to_scout import Label
from .utils import tailf

if TYPE_CHECKING:
    from .to_scout import ScoutQueue


@dataclass
class LabelerDiskQueue:
    scout_queue: ScoutQueue
    mission_dir: Path
    label_queue_size: int = 0

    token_semaphore: threading.BoundedSemaphore = field(init=False, repr=False)

    statistics_lock: threading.Lock = field(default_factory=threading.Lock)
    negatives: int = 0
    positives: int = 0
    total_size: int = 0

    def __post_init__(self) -> None:
        if self.label_queue_size <= 0:
            self.label_queue_size = 9999

        # track number of unlabeled items being handled by the labeler
        self.token_semaphore = threading.BoundedSemaphore(self.label_queue_size)

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
            tile_jpeg.write_bytes(result.data)
            logger.info(f"SAVED TILE {tile_jpeg}")

            # append metadata to unlabeled.jsonl file
            meta_json = result.to_json(index=index)
            with unlabeled_jsonl.open("a") as f:
                f.write(f"{meta_json}\n")

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
                # release next unlabeled result to labeler
                self.token_semaphore.release()

                label = Label.from_json(label_json)
                self.scout_queue.put(label)

                # update stats
                with self.statistics_lock:
                    if label.image_label in [None, "", "0"]:
                        self.negatives += 1
                    else:
                        self.positives += 1
                    self.total_size += label.size
