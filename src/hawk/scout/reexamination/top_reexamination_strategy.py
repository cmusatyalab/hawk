# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import queue
import time
from typing import TYPE_CHECKING

from logzero import logger

from ..core.model import Model
from ..core.object_provider import ObjectProvider
from .reexamination_strategy import ReexaminationStrategy

if TYPE_CHECKING:
    from .reexamination_strategy import ReexaminationQueueType


class TopReexaminationStrategy(ReexaminationStrategy):
    def __init__(self, k: int):
        self._k = k

    @property
    def reexamines_old_results(self) -> bool:
        return True

    def get_new_queues(
        self,
        model: Model,
        old_queues: ReexaminationQueueType,
        start_time: float = 0,
    ) -> tuple[ReexaminationQueueType, int]:
        #new_queue: ReexaminationQueueType = queue.PriorityQueue()

        to_reexamine = []
        num_reexamined = self._k

        for _ in range(num_reexamined):
            try:
                score, time_result, result = old_queues.get_nowait()
                to_reexamine.append((score, time_result, result))
            except queue.Empty:
                break
        if not len(to_reexamine):
            return old_queues, 0

        reexamine = [
            ObjectProvider(item[-1].id, item[-1].content, item[-1].attributes)
            for item in to_reexamine
        ]

        results = model.infer(reexamine)
        for result, prev_result in zip(results, to_reexamine):
            time_result = time.time() - start_time
            obj_id = result.id
            prev_score = prev_result[0]
            score = result.score
            logger.info(
                f"Reexamine score id: {obj_id} "
                f"prev_score{prev_score} "
                f"curr_score {score}"
            )
            #new_queue.put((-score, time_result, result))
            old_queues.put((-score, time_result, result))

        return old_queues, len(reexamine)
        #return new_queue, len(reexamine)
