# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from ..core.result_provider import ResultProvider
from ..reexamination.reexamination_strategy import ReexaminationStrategy
from .topk_selector import TopKSelector


class MaxEntropySelector(TopKSelector):
    def __init__(
        self,
        k: int,
        batch_size: int,
        countermeasure_threshold: float,
        total_countermeasures: int,
        reexamination_strategy: ReexaminationStrategy,
    ):
        super().__init__(
            k,
            batch_size,
            countermeasure_threshold,
            total_countermeasures,
            reexamination_strategy,
        )

    def _add_result(self, result: ResultProvider) -> None:
        assert self._mission is not None

        def calc_score(score: float) -> float:
            return abs(0.5 - score)

        with self._insert_lock:
            time_result = self._mission.mission_time()
            self._priority_queues.put((calc_score(result.score), time_result, result))
            self._batch_added += 1

            if (
                self._batch_added >= self._batch_size
                or self._clear_event.is_set()
                and self._batch_added != 0
            ):
                self.select_tiles(self._k)
