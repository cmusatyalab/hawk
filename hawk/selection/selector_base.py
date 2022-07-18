# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import queue
import threading
from abc import abstractmethod
from typing import Optional, List

from hawk.core.model import Model
from hawk.core.result_provider import ResultProvider
from hawk.selection.selector import Selector
from hawk.context.data_manager_context import DataManagerContext
from hawk.selection.selector_stats import SelectorStats


class SelectorBase(Selector):

    def __init__(self):
        self.result_queue = queue.Queue(maxsize=100)
        self.stats_lock = threading.Lock()
        self.items_processed = 0
        self.num_revisited = 0
        self.num_positives = 0
        self.num_negatives_added = 0
        self.model_train_time = 0
        self.model_examples = 0

        self._model_lock = threading.Lock()
        self._model_present = False
        self._mission = None
        self.transmit_queue = None

        self._finish_event = threading.Event()

    @abstractmethod
    def add_result_inner(self, result: ResultProvider) -> None:
        pass

    @abstractmethod
    def new_model_inner(self, model: Optional[Model]) -> None:
        pass

    def add_easy_negatives(self, path):
        pass

    def add_context(self, context: DataManagerContext):
        self._mission = context

    def add_result(self, result: ResultProvider) -> None:
        with self._model_lock:
            model_present = self._model_present

        if not model_present:
            self.result_queue.put(result)
        else:
            self.add_result_inner(result)

        with self.stats_lock:
            self.items_processed += 1

    def new_model(self, model: Optional[Model]) -> None:
        with self._model_lock:
            self._model_present = model is not None

        self.new_model_inner(model)

    def select_tiles(self):
        pass 
    
    def finish(self) -> None:
        self.select_tiles()
        self._finish_event.set()
        self.result_queue.put(None)

    def get_result(self) -> Optional[ResultProvider]:
        while True:
            try:
                return self.result_queue.get(timeout=10)
            except queue.Empty:
                pass

    def delete_examples(self, examples: List) -> None:
        for path in examples:
            if path.exists():
                path.unlink()

    def get_stats(self) -> SelectorStats:
        with self.stats_lock:
            stats = {'processed_objects': self.items_processed,
                     'items_revisited': self.num_revisited,
                     'negatives_added': self.num_negatives_added,
                     'positive_in_stream': self.num_positives,
                     'train_time': "{:.3f}".format(self.model_train_time),
                     'train_positives': self.model_examples,
                     }

        return SelectorStats(stats)

