# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import queue
from typing import List, Tuple

from hawk.core.model import Model
from hawk.selection.reexamination_strategy import ReexaminationStrategy


class FullReexaminationStrategy(ReexaminationStrategy):

    @property
    def revisits_old_results(self) -> bool:
        return True

    def get_new_queues(self, model: Model, old_queues: List[queue.PriorityQueue], 
                       start_time: float = 0) -> Tuple[List[queue.PriorityQueue], int]:
        new_queue = queue.PriorityQueue()

        to_reexamine = []
        for priority_queue in old_queues:
            while True:
                try:
                    to_reexamine.append(priority_queue.get_nowait()[1])
                except queue.Empty:
                    break

        for result in model.infer(to_reexamine):
            new_queue.put((-result.score, result.id, result))

        return old_queues + [new_queue], len(to_reexamine)
