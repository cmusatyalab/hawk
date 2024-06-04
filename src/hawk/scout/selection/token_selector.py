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
        mission_id: str,
        k: int,
        batch_size: int,
        countermeasure_threshold: float,
        total_countermeasures: int,
        reexamination_strategy: ReexaminationStrategy,
    ):
        super().__init__(
            mission_id,
            k,
            batch_size,
            countermeasure_threshold,
            total_countermeasures,
            reexamination_strategy,
        )
        self.sample_count = 0

    @log_exceptions
    def _initialize_queue(self) -> None:
        assert self._insert_lock.locked()
        for i in range(self._k):
            result = self._priority_queues.get()[-1]
            self.priority_queue_length.dec()

            self.result_queue_length.inc()
            self.result_queue.put(result)
            logger.info(f"Put tile number {i} into result queue.")

    @log_exceptions
    def receive_token_message(self, label: LabelWrapper) -> None:
        # the self._insert_lock is held during reexecution when the priority
        # queues may be replaced with new ones
        with self._insert_lock:
            result = self._priority_queues.get()[-1]
            self.priority_queue_length.dec()

        logger.info(f"New priority queue size is {self._priority_queues.qsize()}")
        self.result_queue_length.inc()
        self.result_queue.put(result)

    def _add_result(self, result: ResultProvider) -> None:
        assert self._mission is not None
        with self._insert_lock:
            self.sample_count += 1

            time_result = self._mission.mission_time()
            self._mission.log(
                f"{self.version} CLASSIFICATION: {result.id} "
                f"GT {result.gt} Score {result.score:.4f}"
            )

            # Queueing positives in stream
            if result.gt:
                logger.info(f"Queueing {result.id} Score {result.score}")

            self.priority_queue_length.inc()
            self._priority_queues.put((-result.score, time_result, result))

            if self._mode == "oracle" and int(result.score) == 1:
                logger.info(f"[Result] Id {result.id} Score {result.score}")

            if self.sample_count % 200 == 0:
                logger.info(f"Total Placed into priority queue: {self.sample_count}")

            if self.sample_count == self._batch_size:
                self._initialize_queue()
