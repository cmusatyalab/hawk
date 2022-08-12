# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import copy
import os
import queue
import time
import threading
from collections import defaultdict
from logzero import logger
from typing import Optional
from pathlib import Path

from hawk.core.model import Model
from hawk.core.result_provider import ResultProvider
from hawk.selection.reexamination_strategy import ReexaminationStrategy
from hawk.selection.selector_base import SelectorBase
from hawk.selection.selector_stats import SelectorStats
from hawk.core.utils import get_example_key, log_exceptions
from hawk.core.utils import ATTR_DATA


class TopKSelector(SelectorBase):

    def __init__(self, k: int, batch_size: int,
                 reexamination_strategy: ReexaminationStrategy,
                 add_negatives: bool = True):
        assert k < batch_size
        super().__init__()

        self.last_result_time = None
        self.timeout = 1000
        self.add_negatives = add_negatives
        self.easy_negatives = defaultdict(list)
        self.version = 0
        self.model_train_time = 0

        self._k = k
        self._batch_size = batch_size
        self._reexamination_strategy = reexamination_strategy
        self._number_excess_batch = 2

        self._priority_queues = [queue.PriorityQueue()]
        self._batch_added = 0
        self._insert_lock = threading.Lock()
        self._mode = "hawk"
        self._en = 0  # 0.1
        
        self.log_counter = [int(i/3.*self._batch_size) for i in range(1, 4)]

    def result_timeout(self, interval=0):
        if interval == 0 or self.last_result_time is None:
            return False
        return (time.time() - self.last_result_time) >= interval

    @log_exceptions
    def select_tiles(self):
        if self._mission.enable_logfile:
            self._mission.log_file.write("{:.3f} {}_{} {}_{} SEL: SELECTION START\n".format(
                time.time() - self._mission.start_time, self._mission.host_name,
                self.version, self._batch_added, self._batch_size))

        for i in range(self._k):
            result = self._priority_queues[-1].get()[-1]
            if self._mission.enable_logfile:
                self._mission.log_file.write("{:.3f} {}_{} {}_{} SEL: FILE SELECTED {}\n".format(
                    time.time() - self._mission.start_time, self._mission.host_name,
                    self.version, i, self._k, result.id))
            if self._mode == "oracle":
                if int(result.score) == 1:
                    self.result_queue.put(result)    
                    logger.info("[Result] Id {} Score {}".format(result.id, result.score))
            else:
                self.result_queue.put(result)
                logger.info("[Result] Id {} Score {}".format(result.id, result.score))
        self._batch_added -= self._batch_size
         
    @log_exceptions
    def add_result_inner(self, result: ResultProvider) -> None:
        with self._insert_lock:

            time_result = time.time() - self._mission.start_time
            self._mission.log_file.write("{:.3f} {}_{} CLASSIFICATION: {} GT {} Score {:.4f}\n".format(
                time_result, self._mission.host_name,
                self.version, result.id, result.obj.gt, result.score))
            if result.obj.gt:
                self.num_positives += 1
                logger.info("Queueing {} Score {}".format(result.id, result.score))
                if self._mission.enable_logfile:
                    self._mission.log_file.write("{:.3f} {}_{} {}_{} INFER: Queue POSITIVE FILE {}\n".format(
                        time_result, self._mission.host_name,
                        self.version, self._batch_added, self._batch_size, result.id))

            self._priority_queues[-1].put((-result.score, time_result, result))
            self._batch_added += 1
            if self._batch_added in self.log_counter:
                logger.info("ADDED {}/{}".format(self._batch_added, self._batch_size))
            if self._batch_added > self._batch_size:
                logger.info("ERROR ADDED {}/{}".format(self._batch_added, self._batch_size))
                
            condition_1 = self._batch_added >= self._batch_size
            condition_2 = self.result_timeout(self.timeout)
            condition_3 = self._finish_event.is_set() and self._batch_added != 0
            if ( condition_1 or condition_2 or condition_3) :
                
                logger.info(f"Sending Results on meeting condition \
                    Buffer full: {condition_1} or Time: {condition_2} or Finish {condition_3}")
                self.last_result_time = time.time()
                self.select_tiles()                    

    @log_exceptions
    def add_easy_negatives(self, path: Path) -> None:
        if not self.add_negatives:
            return

        negative_path = path / '-1'
        os.makedirs(str(negative_path), exist_ok=True)

        result_queue = queue.PriorityQueue()
        with self._insert_lock:
            result_queue.queue = copy.deepcopy(self._priority_queues[-1].queue)

        result_list = [item[-1] for item in list(result_queue.queue)]
        length_results = len(result_list)

        if length_results < 10:
            return
        result_list = sorted(result_list, key=lambda x: x.score, reverse=True)

        num_auto_negative = min(int(0.40 * length_results), 200)
        auto_negative_list = result_list[-num_auto_negative:]

        labels = [1 if '/1/' in item.id else 0 for item in auto_negative_list]
        logger.info("[EASY NEG] Length of result list {} \n \
         negatives added:{} \n ".format(length_results, num_auto_negative, sum(labels)))

        self.num_negatives_added += len(auto_negative_list)

        for result in auto_negative_list:
            object_id = result.id
            example = self._mission.retriever.get_object(object_id, [ATTR_DATA])
            if example is None:
                break
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
                self.version = model.version
                self._mode = model.mode
                self.model_examples = model.train_examples.get('1', 0)
                self.model_train_time = model.train_time
                if self._mode != "hawk":
                    return 
                if version != self.version:
                    versions = [v for v in self.easy_negatives.keys() if v <= version]
                    for v in versions:
                        self.delete_examples(self.easy_negatives[v])

                self._priority_queues, num_revisited = self._reexamination_strategy.get_new_queues(
                    model, self._priority_queues, self._mission.start_time)

                self._batch_added = num_revisited
                logger.info("ADDING  Reexamined to result Queue {}".format(num_revisited))

                self.num_revisited += num_revisited
                self.num_negatives_added = 0
            else:
                # this is a reset, discard everything
                if self._mission.enable_logfile:
                    self._mission.log_file.write("{:.3f} {}_{} ERROR RESET CALLED\n".format(
                        time.time() - self._mission.start_time, self._mission.host_name, self.version))
                self._priority_queues = [queue.PriorityQueue()]
                self._batch_added = 0

            self._mission.log_file.flush()

    def get_stats(self) -> SelectorStats:
        with self.stats_lock:
            stats = {'processed_objects': self.items_processed,
                     'items_revisited': self.num_revisited,
                     'negatives_added': self.num_negatives_added,
                     'positive_in_stream': self.num_positives,
                     'train_positives': self.model_examples,
                     'train_time': "{:.3f}".format(self.model_train_time),
                     }

        return SelectorStats(stats)
