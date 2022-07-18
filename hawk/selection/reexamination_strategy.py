# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import queue
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from hawk.core.model import Model


class ReexaminationStrategy(metaclass=ABCMeta):

    @property
    @abstractmethod
    def revisits_old_results(self) -> bool:
        pass

    @abstractmethod
    def get_new_queues(self, model: Model, 
                       old_queues: List[queue.PriorityQueue]) -> Tuple[List[queue.PriorityQueue], int]:
        pass
