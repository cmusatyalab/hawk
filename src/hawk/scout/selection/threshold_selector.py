# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import queue
import threading
from typing import Optional

from logzero import logger

from ..core.model import Model
from ..core.result_provider import ResultProvider
from ..reexamination.reexamination_strategy import ReexaminationStrategy
from .selector_base import SelectorBase


class ThresholdSelector(SelectorBase):

    def __init__(self, threshold, reexamination_strategy: ReexaminationStrategy):
        super().__init__()

        self._threshold = threshold
        self._reexamination_strategy = reexamination_strategy

        self._discard_queue = [queue.PriorityQueue()]
        self._insert_lock = threading.Lock()
        self._items_dropped = 0
        self._false_negatives = 0

    def _add_result(self, result: ResultProvider) -> None:
        if result.gt:
           self.num_positives += 1
           logger.info("{} Score {}".format(result.id, result.score))

        if result.score > self._threshold:
            self.result_queue.put(result)
        elif self._reexamination_strategy.reexamines_old_results:
            with self._insert_lock:
                self._discard_queue[-1].put((-result.score, result.id, result))
        else:
            with self.stats_lock:
                self._items_dropped += 1
                if result.gt:
                    self._false_negatives += 1

    def _new_model(self, model: Optional[Model]) -> None:
        with self._insert_lock:
            if model is not None:
                version = self.version
                self.version = model.version
                self.model_examples = model.train_examples.get('1', 0)
                self._priority_queues, num_revisited = self._reexamination_strategy.get_new_queues(
                    model, self._priority_queues, self._mission.start_time)

                self.num_revisited += num_revisited
