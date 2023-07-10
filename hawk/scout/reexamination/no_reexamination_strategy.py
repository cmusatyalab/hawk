# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import queue
from typing import List, Tuple

from ..core.model import Model
from .reexamination_strategy import ReexaminationStrategy


class NoReexaminationStrategy(ReexaminationStrategy):

    @property
    def reexamines_old_results(self) -> bool:
        return False

    def get_new_queues(self, model: Model, old_queues: List[queue.PriorityQueue], 
                       start_time: float = 0) -> Tuple[List[queue.PriorityQueue], int]:
        return old_queues + [queue.PriorityQueue()], 0
