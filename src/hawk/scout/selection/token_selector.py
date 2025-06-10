# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from logzero import logger

from ...classes import NEGATIVE_CLASS
from ...proto.messages_pb2 import SendLabel
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
        sliding_window: bool,
        upper_threshold_start: float,
        upper_threshold_delta: float,
        lower_threshold_start: float,
        lower_threshold_delta: float,
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
        self.sliding_window = sliding_window
        self.upper_threshold_start = upper_threshold_start
        self.upper_threshold_delta = upper_threshold_delta
        self.lower_threshold_start = lower_threshold_start
        self.lower_threshold_delta = lower_threshold_delta
        self.upper_threshold = self.upper_threshold_start
        logger.info(
            f"Token attrs: {self.sliding_window}, "
            f"{self.upper_threshold_delta}, {self.upper_threshold_start}, "
            f"{self.lower_threshold_delta}, {self.lower_threshold_start}"
        )

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
    def receive_token_message(self, label: SendLabel) -> None:
        # the self._insert_lock is held during reexecution when the priority
        # queues may be replaced with new ones
        with self._insert_lock:
            if self.sliding_window:
                # find first sample below the current upper threshold for sliding window
                result = self.find_result()
            else:
                result = self._priority_queues.get()[-1]
            self.priority_queue_length.dec()

        self.result_queue_length.inc()
        self.result_queue.put(result)

    def _add_result(self, result: ResultProvider) -> None:
        # logger.info("In Token add result...")
        assert self._mission is not None
        with self._insert_lock:
            self.sample_count += 1

            time_result = self._mission.mission_time()
            self._mission.log(
                f"{self.version} CLASSIFICATION: {result.id} "
                f"GT {result.gt} Score {result.score:.4f}"
            )

            # Queueing positives in stream
            if result.gt != NEGATIVE_CLASS:
                logger.info(f"Queueing {result.id} Score {result.score}")

            self.priority_queue_length.inc()
            self._priority_queues.put((-result.score, time_result, result))

            if self._mode == "oracle" and int(result.score) == 1:
                logger.info(f"[Result] Id {result.id} Score {result.score}")

            if self.sample_count % 200 == 0:
                logger.info(f"Total Placed into priority queue: {self.sample_count}")

            if self.sample_count == self._batch_size:
                self._initialize_queue()

    def find_result(self) -> ResultProvider:
        ## set upper limit on scores
        self.upper_threshold = self.upper_threshold_start * (
            self.upper_threshold_delta**self.version
        )
        # logger.info(f"In find result, upper threshold = {self.upper_threshold}")
        temp_item_list: list[tuple[float, float, ResultProvider]] = []
        while not self._priority_queues.empty():
            item = self._priority_queues.get()
            score = item[0] * -1
            if score <= self.upper_threshold:
                ## put all temp items back in the queue and return the current
                ## item, and return current
                for temp_item in temp_item_list:
                    self._priority_queues.put(temp_item)
                # logger.info(f"Returning item with score: {item[0]}")
                return item[-1]

            # if score is greater than upper thresh, append to temp list and
            # keep looping
            temp_item_list.append(item)

        ## if all samples in pri queue above upper thresh.
        for temp_item in temp_item_list:
            self._priority_queues.put(temp_item)
        return self._priority_queues.get()[-1]
