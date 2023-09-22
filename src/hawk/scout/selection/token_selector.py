# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import time

from logzero import logger

from ..core.result_provider import ResultProvider
from ..core.utils import log_exceptions
from ..reexamination.reexamination_strategy import ReexaminationStrategy
from .topk_selector import TopKSelector


class TokenSelector(TopKSelector):
    def __init__(
        self, k: int, batch_size: int, reexamination_strategy: ReexaminationStrategy
    ):
        super().__init__(k, batch_size, reexamination_strategy)
        self.sample_count = 0

    @log_exceptions
    def _initialize_queue(self):
        if self._mode == "oracle":
            return

        for i in range(self._k):
            result = self._priority_queues[-1].get()[-1]
            self.result_queue.put(result)
            logger.info(f"Put tile number {self.sample_count} into result queue.")

    @log_exceptions
    def receive_token_message(self, label):
        logger.info("In receive token message in token selector...")
        logger.info(
            "Index and label of received label: {} ... {} \n".format(
                label.scoutIndex, label.imageLabel
            )
        )
        result = self._priority_queues[-1].get()[-1]
        self.result_queue.put(result)
        logger.info("Sent new sample as a result of token message...")

    def _add_result(self, result: ResultProvider) -> None:
        self.sample_count += 1
        with self._insert_lock:
            time_result = time.time() - self._mission.start_time
            self._mission.log_file.write(
                "{:.3f} {}_{} CLASSIFICATION: {} GT {} Score {:.4f}\n".format(
                    time_result,
                    self._mission.host_name,
                    self.version,
                    result.id,
                    result.gt,
                    result.score,
                )
            )

            # Incrementing positives in stream
            if result.gt:
                self.num_positives += 1
                logger.info(f"Queueing {result.id} Score {result.score}")

            if self._mode == "oracle":
                if int(result.score) == 1:
                    self.result_queue.put(result)
                    logger.info(f"[Result] Id {result.id} Score {result.score}")

            self._priority_queues[-1].put((-result.score, time_result, result))
            # pop the top 4 samples of the first 100 to populate the initial labeling queue at home.
            # logger.info("Self.k parameter: {}".format(self._k))
            if self.sample_count == 1000:
                self._initialize_queue()
