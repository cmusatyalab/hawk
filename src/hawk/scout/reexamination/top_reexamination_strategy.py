# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import multiprocessing as mp
import queue
import time
from typing import List, Tuple

from logzero import logger

from ..core.model import Model
from ..core.object_provider import ObjectProvider
from .reexamination_strategy import ReexaminationStrategy


class TopReexaminationStrategy(ReexaminationStrategy):
    def __init__(self, k: int):
        self._k = k
        self.reexamined = mp.Queue()

    @property
    def reexamines_old_results(self) -> bool:
        return True

    def get_new_queues(
        self, model: Model, old_queues: List[queue.PriorityQueue], start_time: float = 0
    ) -> Tuple[List[queue.PriorityQueue], int]:
        new_queue = queue.PriorityQueue()  # To start a new priority queue
        # new_queue =  old_queues[-1] # Reuse the same queue

        to_reexamine = []
        num_queues = len(old_queues)
        num_reexamined = []

        mod_ = self._k % num_queues
        num_per_queue = self._k // num_queues

        for i in range(num_queues):
            if i < mod_:
                num_reexamined.append(int(num_per_queue + 1))
            else:
                num_reexamined.append(int(num_per_queue))

        for priority_queue, num_examine in zip(old_queues, num_reexamined):
            for _ in range(num_examine):
                try:
                    score, time_result, result = priority_queue.get_nowait()
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
            new_queue.put((-score, time_result, result))

        return old_queues + [new_queue], len(reexamine)
