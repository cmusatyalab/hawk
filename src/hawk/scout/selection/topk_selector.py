# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import copy
import os
import queue
import random
import threading
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from logzero import logger

from ..core.model import Model
from ..core.result_provider import ResultProvider
from ..core.utils import get_example_key, log_exceptions
from .selector_base import SelectorBase

if TYPE_CHECKING:
    from ..reexamination.reexamination_strategy import (
        ReexaminationQueueType,
        ReexaminationStrategy,
    )


class TopKSelector(SelectorBase):
    def __init__(
        self,
        mission_id: str,
        k: int,
        batch_size: int,
        countermeasure_threshold: float,
        total_countermeasures: int,
        reexamination_strategy: ReexaminationStrategy,
        add_negatives: bool = True,
    ):
        logger.info(f"K: {k}, batchsize: {batch_size}")
        assert k < batch_size
        super().__init__(mission_id)

        self.version = 0
        self.add_negatives = add_negatives
        self.easy_negatives: dict[int, list[Path]] = defaultdict(list)
        self.num_negatives_added = 0

        self._k = k
        self._batch_size = batch_size
        self.countermeasure_threshold = countermeasure_threshold
        self.num_countermeasures = total_countermeasures
        self._reexamination_strategy = reexamination_strategy

        self._priority_queues: ReexaminationQueueType = queue.PriorityQueue()
        self._batch_added = 0
        self._insert_lock = threading.Lock()
        self._mode = "hawk"

        self.log_counter = [int(i / 3.0 * self._batch_size) for i in range(1, 4)]

    @log_exceptions
    def select_tiles(self, num_tiles: int) -> None:
        assert self._mission is not None
        for i in range(num_tiles):
            result = self._priority_queues.get()[-1]
            self.priority_queue_length.dec()
            self._mission.log(
                f"{self.version} {i}_{self._k} SEL: FILE SELECTED {result.id}"
            )
            if self._mode != "oracle":
                self.result_queue_length.inc()
                self.result_queue.put(result)
                logger.info(f"[Result] Id {result.id} Score {result.score}")
        self._batch_added -= self._batch_size

    @log_exceptions
    def _add_result(self, result: ResultProvider) -> None:
        assert self._mission is not None
        with self._insert_lock:
            time_result = self._mission.mission_time()
            self._mission.log(
                f"{self.version} CLASSIFICATION: {result.id} "
                f"GT {result.gt} Score {result.score:.4f}"
            )

            # Incrementing positives in stream
            if result.gt:
                logger.info(f"Queueing {result.id} Score {result.score}")

            if self._mode == "oracle" and int(result.score) == 1:
                self.result_queue_length.inc()
                self.result_queue.put(result)
                logger.info(f"[Result] Id {result.id} Score {result.score}")

            self.priority_queue_length.inc()
            ## need adjustment here to account for different simple methods for
            ## prioritizing sample, by default is sum of all pos classes
            self._priority_queues.put((-result.score, time_result, result))
            self._batch_added += 1

            # Logging for debugging
            if self._batch_added in self.log_counter:
                logger.info(f"ADDED {self._batch_added}/{self._batch_size}")

            if self._batch_added >= self._batch_size or (
                self._clear_event.is_set() and self._batch_added != 0
            ):
                logger.info("Select tiles being called in add_result topk...")
                self.select_tiles(self._k)

    @log_exceptions
    def add_easy_negatives(self, path: Path) -> None:
        if not self.add_negatives:
            return

        assert self._mission is not None
        negative_path = path / "-1"
        os.makedirs(str(negative_path), exist_ok=True)

        result_queue: ReexaminationQueueType = queue.PriorityQueue()
        with self._insert_lock:
            result_queue.queue = copy.deepcopy(self._priority_queues.queue)

        result_list = [item[-1] for item in list(result_queue.queue)]
        length_results = len(result_list)

        if length_results < 10:
            return

        num_auto_negative = min(int(0.40 * length_results), 200)
        # result_list = sorted(result_list, key=lambda x: x.score)
        # auto_negative_list = result_list[:num_auto_negative]
        auto_negative_list = random.sample(result_list, num_auto_negative)

        labels = [0 if item.id.startswith("/0/") else 1 for item in auto_negative_list]
        logger.info(
            f"[EASY NEG] Length of result list {length_results}"
            f" negatives added: {num_auto_negative}"
            f" total false negatives: {sum(labels)}"
        )

        self.num_negatives_added += len(auto_negative_list)

        for result in auto_negative_list:
            example_data = result.read_object(self._mission.retriever)
            if example_data is None:
                continue

            example_file = get_example_key(example_data)
            example_path = negative_path.joinpath(example_file)
            example_path.write_bytes(example_data)
            with self._insert_lock:
                self.easy_negatives[self.version].append(example_path)

    def delete_examples(self, examples: list[Path]) -> None:
        for path in examples:
            if path.exists():
                path.unlink()

    def _new_model(self, model: Model | None) -> None:
        assert self._mission is not None
        with self._insert_lock:
            if model is not None:
                version = self.version
                self.version = model.version
                self._mode = model.mode
                self.model_examples = model.train_examples.get("1", 0)
                if version != self.version:
                    versions = [v for v in self.easy_negatives if v <= version]
                    for v in versions:
                        self.delete_examples(self.easy_negatives[v])

                (
                    self._priority_queues,
                    num_revisited,
                ) = self._reexamination_strategy.get_new_queues(
                    model,
                    self._priority_queues,
                    self._mission.start_time,
                )

                self.priority_queue_length.set(self._priority_queues.qsize())
                self.num_revisited.inc(num_revisited)

                self._batch_added += num_revisited
                logger.info(f"ADDING Reexamined to result Queue {num_revisited}")

                self.num_negatives_added = 0
            else:
                # this is a reset, discard everything
                self._priority_queues = queue.PriorityQueue()
                self.priority_queue_length.set(0)
                self._batch_added = 0
