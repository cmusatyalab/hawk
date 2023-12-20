# SPDX-FileCopyrightText: 2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import queue
from dataclasses import dataclass
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    MetaQueueType = Queue[str]
    LabelQueueType = Queue[str]
else:
    MetaQueueType = Queue
    LabelQueueType = Queue


@dataclass
class LabelStats:
    positives: int = 0  # labeled true positives
    negatives: int = 0  # labeled false positives
    total_bytes: int = 0  # sum of meta_data["size"]

    def __post_init__(self) -> None:
        self.queue: Queue[tuple[int, int, int]] = Queue()

    def update(self, positives: int, negatives: int, total_bytes: int) -> None:
        """Used in another thread to queue updated values"""
        self.queue.put((positives, negatives, total_bytes))

    def resync(self) -> None:
        """Read any queued values to update to the latest available"""
        try:
            while True:
                (
                    self.positives,
                    self.negatives,
                    self.total_bytes,
                ) = self.queue.get_nowait()
        except queue.Empty:
            pass


class Labeler:
    def start_labeling(
        self,
        input_q: MetaQueueType,
        result_q: LabelQueueType,
        labelstats: LabelStats,
        stop_event: Event,
    ) -> None:
        ...
