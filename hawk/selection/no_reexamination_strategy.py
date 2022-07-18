# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import queue
from typing import List, Tuple

from hawk.core.model import Model
from hawk.selection.reexamination_strategy import ReexaminationStrategy


class NoReexaminationStrategy(ReexaminationStrategy):

    @property
    def revisits_old_results(self) -> bool:
        return False

    def get_new_queues(self, model: Model, old_queues: List[queue.PriorityQueue], 
                       start_time: float = 0) -> Tuple[List[queue.PriorityQueue], int]:
        return [queue.PriorityQueue()], 0
