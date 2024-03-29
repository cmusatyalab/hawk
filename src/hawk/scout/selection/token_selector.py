# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from logzero import logger

from ...proto.messages_pb2 import LabelWrapper
from ..core.result_provider import ResultProvider
from ..core.utils import log_exceptions
from ..reexamination.reexamination_strategy import ReexaminationStrategy
from .topk_selector import TopKSelector


class TokenSelector(TopKSelector):
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
        self.sample_count = 0

    @log_exceptions
    def _initialize_queue(self) -> None:
        for _i in range(self._k):
            result = self._priority_queues.get()[-1]
            self.result_queue.put(result)
            logger.info(f"Put tile number {self.sample_count} into result queue.")

    @log_exceptions
    def receive_token_message(self, label: LabelWrapper) -> None:
        result = self._priority_queues.get()[-1]
        self.result_queue.put(result)

    def _add_result(self, result: ResultProvider) -> None:
        assert self._mission is not None
        self.sample_count += 1
        with self._insert_lock:
            time_result = self._mission.mission_time()
            self._mission.log(
                f"{self.version} CLASSIFICATION: {result.id} "
                f"GT {result.gt} Score {result.score:.4f}"
            )

            # Incrementing positives in stream
            if result.gt:
                self.num_positives += 1
                logger.info(f"Queueing {result.id} Score {result.score}")

            if self._mode == "oracle":
                self._priority_queues.put((-result.score, time_result, result))
                if int(result.score) == 1:
                    logger.info(f"[Result] Id {result.id} Score {result.score}")
            else:
                self._priority_queues.put((-result.score, time_result, result))

        if self.sample_count % 200 == 0:
            logger.info(f"Total Placed into priority queue: {self.sample_count}")

        if self.sample_count == self._batch_size:
            self._initialize_queue()
