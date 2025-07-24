# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

from logzero import logger

from ...classes import NEGATIVE_CLASS
from ..stats import (
    HAWK_SELECTOR_DISCARD_QUEUE_LENGTH,
    HAWK_SELECTOR_DROPPED_OBJECTS,
    HAWK_SELECTOR_FALSE_NEGATIVES,
)
from .selector_base import SelectorBase

if TYPE_CHECKING:
    from pathlib import Path

    from ..core.model import Model
    from ..core.result_provider import ResultProvider
    from ..reexamination.reexamination_strategy import (
        ReexaminationQueueType,
        ReexaminationStrategy,
    )


class ThresholdSelector(SelectorBase):
    def __init__(
        self,
        mission_id: str,
        threshold: float,
        reexamination_strategy: ReexaminationStrategy,
    ) -> None:
        super().__init__(mission_id)

        self._threshold = threshold
        self._reexamination_strategy = reexamination_strategy

        self.discard_queue_length = HAWK_SELECTOR_DISCARD_QUEUE_LENGTH.labels(
            mission=mission_id,
        )
        self._discard_queue: ReexaminationQueueType = queue.PriorityQueue()
        self._insert_lock = threading.Lock()
        self.items_dropped = HAWK_SELECTOR_DROPPED_OBJECTS.labels(mission=mission_id)
        self.false_negatives = HAWK_SELECTOR_FALSE_NEGATIVES.labels(mission=mission_id)

    def _add_result(self, result: ResultProvider) -> None:
        if result.gt != NEGATIVE_CLASS:
            logger.info(f"{result.id} Score {result.score}")

        if result.score > self._threshold:
            self.result_queue_length.inc()
            self.result_queue.put(result)
        elif self._reexamination_strategy.reexamines_old_results:
            assert self._mission is not None
            with self._insert_lock:
                time_result = self._mission.mission_time()
                self.discard_queue_length.inc()
                self._discard_queue.put((-result.score, time_result, result))
        else:
            self.items_dropped.inc()
            if result.gt != NEGATIVE_CLASS:
                self.false_negatives.inc()

    def _new_model(self, model: Model | None) -> None:
        assert self._mission is not None
        with self._insert_lock:
            if model is not None:
                # version = self.version
                self.version = model.version
                self.model_examples = model.train_examples.get("1", 0)
                (
                    self._discard_queue,
                    num_revisited,
                ) = self._reexamination_strategy.get_new_queues(
                    model,
                    self._discard_queue,
                    self._mission.start_time,
                )
                self.discard_queue_length.set(self._discard_queue.qsize())
                self.num_revisited.inc(num_revisited)

    def add_easy_negatives(self, path: Path) -> None:
        """Not implemented yet."""
