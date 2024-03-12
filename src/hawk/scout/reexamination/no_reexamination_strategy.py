# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import queue
from typing import TYPE_CHECKING

from ..core.model import Model
from .reexamination_strategy import ReexaminationStrategy

if TYPE_CHECKING:
    from .reexamination_strategy import ReexaminationQueueType


class NoReexaminationStrategy(ReexaminationStrategy):
    @property
    def reexamines_old_results(self) -> bool:
        return False

    def get_new_queues(
        self,
        model: Model,
        old_queues: list[ReexaminationQueueType],
        start_time: float = 0,
    ) -> tuple[list[ReexaminationQueueType], int]:
        return old_queues + [queue.PriorityQueue()], 0
