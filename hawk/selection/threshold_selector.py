# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import queue
import threading
from logzero import logger
from typing import Optional

from hawk.core.model import Model
from hawk.core.result_provider import ResultProvider
from hawk.selection.reexamination_strategy import ReexaminationStrategy
from hawk.selection.selector_base import SelectorBase
from hawk.selection.selector_stats import SelectorStats


class ThresholdSelector(SelectorBase):

    def __init__(self, threshold, reexamination_strategy: ReexaminationStrategy):
        super().__init__()

        self._threshold = threshold
        self._reexamination_strategy = reexamination_strategy

        self._discard_queue = [queue.PriorityQueue()]
        self._insert_lock = threading.Lock()
        self._items_dropped = 0
        self._false_negatives = 0

    def add_result_inner(self, result: ResultProvider) -> None:
        if result.obj.gt:
           logger.info("{} Score {}".format(result.id, result.score))

        if result.score > self._threshold:
            self.result_queue.put(result)
        elif self._reexamination_strategy.revisits_old_results:
            with self._insert_lock:
                self._discard_queue[-1].put((-result.score, result.id, result))
        else:
            with self.stats_lock:
                self._items_dropped += 1
                if result.obj.gt:
                    self._false_negatives += 1

    def new_model_inner(self, model: Optional[Model]) -> None:
        if not self._reexamination_strategy.revisits_old_results:
            return

        with self._insert_lock:
            if model is not None:
                self._discard_queue = self._reexamination_strategy.get_new_queues(model, self._discard_queue)
            else:
                # this is a reset, discard everything
                self._discard_queue = [queue.PriorityQueue()]

    def get_stats(self) -> SelectorStats:
        with self.stats_lock:
            items_processed = self.items_processed
            items_dropped = self._items_dropped
            false_negatives = self._false_negatives

        return SelectorStats(items_processed,
                             items_dropped,
                             items_processed - items_dropped if not self._reexamination_strategy.revisits_old_results else None,
                             false_negatives)
