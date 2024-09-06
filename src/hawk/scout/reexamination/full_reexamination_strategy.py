# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
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


class FullReexaminationStrategy(ReexaminationStrategy):
    @property
    def reexamines_old_results(self) -> bool:
        return True

    def get_new_queues(
        self,
        model: Model,
        old_queue: ReexaminationQueueType,
        start_time: float = 0,
    ) -> tuple[ReexaminationQueueType, int]:
        new_queue: ReexaminationQueueType = queue.PriorityQueue()

        to_reexamine = []
        try:
            while True:
                score, time_result, result = old_queue.get_nowait()
                to_reexamine.append((score, time_result, result))
        except queue.Empty:
            pass

        reexamine = [
            obj
            for _, _, result in to_reexamine
            if (obj := ObjectProvider.from_result_provider(result, self.retriever))
            is not None
        ]

        results = model.infer(reexamine)

        for result, prev_result in zip(results, to_reexamine):
            time_result = time.time() - start_time
            obj_id = result.id
            prev_score = prev_result[0]
            score = result.score
            logger.info(
                f"Reexamine score id: {obj_id} "
                f"prev_score{prev_score} curr_score {score}"
            )
            new_queue.put((-score, time_result, result))

        return new_queue, len(to_reexamine)
