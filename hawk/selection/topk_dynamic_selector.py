# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import copy
import os
import math
import queue
import time
import threading
from collections import defaultdict
from logzero import logger
from typing import Optional, Dict
from pathlib import Path

from hawk.core.model import Model
from hawk.core.result_provider import ResultProvider
from hawk.selection.reexamination_strategy import ReexaminationStrategy
from hawk.selection.selector_base import SelectorBase
from hawk.selection.selector_stats import SelectorStats
from hawk.core.utils import get_example_key
from hawk.core.utils import ATTR_DATA
from hawk.core.utils import log_exceptions


class TopKDynamicSelector(SelectorBase):

    def __init__(self, batch_size: int, params: Dict,
                 reexamination_strategy: ReexaminationStrategy,
                 add_negatives: bool = True):
        super().__init__()

        self._batch_size = batch_size
        self._reexamination_strategy = reexamination_strategy
        self._number_excess_batch = 2

        self._priority_queues = [queue.PriorityQueue()]
        self._batch_added = 0
        self._insert_lock = threading.Lock()
        self.last_result_time = None
        self.timeout = 300
        self.add_negatives = add_negatives
        self.easy_negatives = defaultdict(list)
        self.version = -1

        self.k_max = params.get('k_max', 8)
        self.k_min = params.get('k_min', 2)
        self.k_slope = params.get('k_slope', 0.05)
        self.p_max = params.get('p_max', 0.999)
        self.p_min = params.get('p_min', 0.8)
        self.p_slope = params.get('p_slope', 0.01)
        self.en_max = params.get('en_max', 0.3)
        self.en_min = params.get('en_min', 0.05)
        self.en_slope = params.get('en_slope', 0.05)

        self._k = self.k_max
        self._p = self.p_max
        self._en = self.en_min

        self.num_negatives_added = 0
        self.num_revisited = 0
        self.num_positives = 0


    def get_current_k(self, round: int):
        return max(self.k_min, self.k_max * (1 - self.k_slope)**round)

    def get_current_p(self, round: int):
        return max(self.p_min, self.p_max * (1 - self.p_slope)**round)

    def get_current_en(self, round: int):
        return min(self.en_max, self.en_min * (1 + self.en_slope)**round)

    def result_timeout(self, interval=0):
        if interval == 0 or self.last_result_time is None:
            return False
        return (time.time() - self.last_result_time) >= interval

    @log_exceptions
    def add_result_inner(self, result: ResultProvider) -> None:
        with self._insert_lock:
            assert result.score >= 0 and result.score <= 1, "Score not a probability"
            path_name = ('/').join(result.id.split('/')[-2:])
            if result.obj.gt:
                self.num_positives += 1
                logger.info("Queueing {} Score {}".format(path_name, result.score))

            self._priority_queues[-1].put((-result.score, 255-ord(path_name[0]), result.id,  result))
            self._batch_added += 1
            # if self._batch_added == self._batch_size or \
            #     (self.result_timeout(self.timeout) and self._batch_added > 0):
            if self._batch_added == self._batch_size:
                # k items + if item's score is above p_threshold
                logger.info("ROUND {} \n P {} \n K {} \n EN {}".format(self.version, self._p, self._k, self._en))
                count = 0
                while True:
                    count += 1
                    result = self._priority_queues[-1].get()[-1]
                    if result.score <= self._en:
                        logger.info("[Result] LOWER THAN EN Id {} Score {}".format(result.id, result.score))
                        break
                    logger.info("[Result] Id {} Score {}".format(result.id, result.score))
                    self.result_queue.put(result)
                    if not (count <= self._k or result.score >= self._p):
                        break

                self._batch_added = 0
                self.last_result_time = time.time()

    @log_exceptions
    def add_easy_negatives(self, path: Path) -> None:
        if not self.add_negatives:
            return

        negative_path = path / '-1'
        os.makedirs(str(negative_path), exist_ok=True)

        result_list = []
        result_queue = queue.PriorityQueue()
        with self._insert_lock:
            result_queue.queue = copy.deepcopy(self._priority_queues[-1].queue)

        result_list = [item[-1] for item in list(result_queue.queue)]
        length_results = len(result_list)

        if length_results <= self.k_max:
            return

        result_list = sorted(result_list, key= lambda x: x.score, reverse=True)

        num_auto_negative = int(0.25 * length_results)
        auto_negative_index = length_results - num_auto_negative
        score = result_list[auto_negative_index].score

        # Ensure easy negatives below en_threshold
        while score >= self._en:
            auto_negative_index += 1
            score = result_list[auto_negative_index].score

        auto_negative_list = result_list[auto_negative_index:]
        num_auto_negative = len(auto_negative_list)
        logger.info("[EASY NEG] Length of result list {} {}".format(length_results, num_auto_negative))
        self.num_negatives_added += num_auto_negative

        for result in auto_negative_list:
            object_id = result.id
            example = self._mission.retriever.get_object(object_id, [ATTR_DATA])
            example_file = get_example_key(example.content)
            example_path = negative_path / example_file
            with example_path.open('wb') as f:
                f.write(example.content)
            with self._insert_lock:
                self.easy_negatives[self.version].append(example_path)

    def new_model_inner(self, model: Optional[Model]) -> None:
        with self._insert_lock:
            if model is not None:
                version = self.version
                if version in self.easy_negatives:
                    versions = [v for v in self.easy_negatives.keys() if v <= version]
                    for v in versions:
                        self.delete_examples(self.easy_negatives[v])

                # self.num_negatives_added = 0
                self.version += 1
                self._k = self.get_current_k(self.version)
                self._p = self.get_current_p(self.version)
                self._en = self.get_current_en(self.version)
                # add fractional batch before possibly discarding results in old queue
                logger.info("ADDING some to result Queue {}".format(math.ceil(float(self._k) * self._batch_added / self._batch_size)))
                for _ in range(math.ceil(float(self._k) * self._batch_added / self._batch_size)):
                    self.result_queue.put(self._priority_queues[-1].get()[-1])
                self._priority_queues = self._reexamination_strategy.get_new_queues(model, self._priority_queues)
                self.num_revisited += len(list(self._priority_queues[-1].queue))
            else:
                # this is a reset, discard everything
                self._priority_queues = [queue.PriorityQueue()]

            self._batch_added = 0

    def get_stats(self) -> SelectorStats:
        with self.stats_lock:
            stats = {'processed_objects': self.items_processed,
                     'items_revisited': self.num_revisited,
                     'negatives_added': self.num_negatives_added,
                     'positive_in_stream': self.num_positives,
                    }

        return SelectorStats(stats)
