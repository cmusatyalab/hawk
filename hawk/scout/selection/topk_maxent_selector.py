# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import math
import queue
import threading
import time
from typing import Optional

from ..core.model import Model
from ..core.result_provider import ResultProvider
from ..reexamination.reexamination_strategy import ReexaminationStrategy
from .selector_base import SelectorBase, SelectorStats
from .topk_selector import TopKSelector


class MaxEntropySelector(TopKSelector):

    def __init__(self, k: int, batch_size: int, reexamination_strategy: ReexaminationStrategy):
        super().__init__(k, batch_size, reexamination_strategy)


    def _add_result(self, result: ResultProvider) -> None:
        def calc_score(score):
            return abs(0.5 - score)

        with self._insert_lock:
            time_result = time.time() - self._mission.start_time
            self._priority_queues[-1].put((calc_score(result.score), time_result, result))
            self._batch_added += 1
            
            if (self._batch_added >= self._batch_size or 
                self._clear_event.is_set() and self._batch_added != 0) :
                self.select_tiles(self._k) 
