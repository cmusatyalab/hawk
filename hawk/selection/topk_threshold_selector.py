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


class TopKThresholdSelector(SelectorBase):

    def __init__(self, k: int, batch_size: int, 
                 reexamination_strategy: ReexaminationStrategy):
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
        self.threshold = 0.4
        self.threshold_increment = 0.05
        self.threshold_max = 0.85
        self.model_version = 0

    def result_timeout(self, interval=0):
        if interval == 0 or self.last_result_time is None:
            return False
        return (time.time() - self.last_result_time) >= interval

    def add_result_inner(self, result: ResultProvider) -> None:
        with self._insert_lock:
            self._priority_queues[-1].put((-result.score, result.id, result))
            self._batch_added += 1
            if self._batch_added == self._batch_size:
                count = 0
                while count < self._k:
                    score, _, result = self._priority_queues[-1].get()
                    if score >= self.threshold:
                        count += 1
                        self.result_queue.put(result)
                self._batch_added = 0
                self.last_result_time = time.time()

    def new_model_inner(self, model: Optional[Model]) -> None:
        with self._insert_lock:
            if model is not None:
                self.model_version += 1
                if self.model_version % 4 == 0:
                    self.threshold = min(self.threshold_max, self.threshold + self.threshold_increment)
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
