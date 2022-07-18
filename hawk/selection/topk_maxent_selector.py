# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import math
import queue
import time
import threading
from typing import Optional

from hawk.core.model import Model
from hawk.core.result_provider import ResultProvider
from hawk.selection.reexamination_strategy import ReexaminationStrategy
from hawk.selection.selector_base import SelectorBase
from hawk.selection.selector_stats import SelectorStats


class MaxEntropySelector(SelectorBase):

    def __init__(self, k: int, batch_size: int, reexamination_strategy: ReexaminationStrategy):
        assert k < batch_size
        super().__init__()

        self._k = k
        self._batch_size = batch_size
        self._reexamination_strategy = reexamination_strategy

        self._priority_queues = [queue.PriorityQueue()]
        self._batch_added = 0
        self._insert_lock = threading.Lock()
        self.last_result_time = None
        self.timeout = 300

    def result_timeout(self, interval=0):
        if interval == 0 or self.last_result_time is None:
            return False
        return (time.time() - self.last_result_time) >= interval

    def add_result_inner(self, result: ResultProvider) -> None:
        def calc_score(score):
            return abs(0.5 - score)

        with self._insert_lock:
            self._priority_queues[-1].put((calc_score(result.score), result.id, result))
            self._batch_added += 1
            # if self._batch_added == self._batch_size or \
            #     (self.result_timeout(self.timeout) and self._batch_added > 0):
            if self._batch_added == self._batch_size:
                for _ in range(self._k):
                    self.result_queue.put(self._priority_queues[-1].get()[-1])
                self._batch_added = 0
                self.last_result_time = time.time()

    def new_model_inner(self, model: Optional[Model]) -> None:
        with self._insert_lock:
            if model is not None:
                # add fractional batch before possibly discarding results in old queue
                for _ in range(math.ceil(float(self._k) * self._batch_added / self._batch_size)):
                    self.result_queue.put(self._priority_queues[-1].get()[-1])
                self._priority_queues = self._reexamination_strategy.get_new_queues(model, self._priority_queues)
            else:
                # this is a reset, discard everything
                self._priority_queues = [queue.PriorityQueue()]

            self._batch_added = 0

    def get_stats(self) -> SelectorStats:
        with self.stats_lock:
            items_processed = self.items_processed

        return SelectorStats(items_processed, 0, None, 0)
