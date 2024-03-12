# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import queue
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Tuple

from ..core.model import Model
from ..core.result_provider import ResultProvider

if TYPE_CHECKING:
    ReexaminationQueueType = queue.PriorityQueue[Tuple[float, float, ResultProvider]]


class ReexaminationStrategy(metaclass=ABCMeta):
    @property
    @abstractmethod
    def reexamines_old_results(self) -> bool:
        """Returns True if old results are reexamined by strategy"""
        pass

    @abstractmethod
    def get_new_queues(
        self,
        model: Model,
        old_queues: ReexaminationQueueType,
        start_time: float = 0,
    ) -> tuple[ReexaminationQueueType, int]:
        """Generates a new queue with reexamined results"""
        pass
