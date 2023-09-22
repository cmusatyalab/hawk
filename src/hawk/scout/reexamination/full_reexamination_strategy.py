# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import queue
import time
from typing import List, Tuple

from logzero import logger

from ..core.model import Model
from ..core.object_provider import ObjectProvider
from .reexamination_strategy import ReexaminationStrategy


class FullReexaminationStrategy(ReexaminationStrategy):
    @property
    def reexamines_old_results(self) -> bool:
        return True

    def get_new_queues(
        self, model: Model, old_queues: List[queue.PriorityQueue], start_time: float = 0
    ) -> Tuple[List[queue.PriorityQueue], int]:
        new_queue = queue.PriorityQueue()

        to_reexamine = []
        for priority_queue in old_queues:
            while True:
                try:
                    score, time_result, result = priority_queue.get_nowait()
                    to_reexamine.append((score, time_result, result))
                except queue.Empty:
                    break

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
                f"Reexamine score id: {obj_id} prev_score{prev_score} curr_score {score}"
            )
            new_queue.put((-score, time_result, result))

        for result in model.infer(to_reexamine):
            new_queue.put((-result.score, result.id, result))

        return old_queues + [new_queue], len(to_reexamine)
