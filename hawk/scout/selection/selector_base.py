# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import queue
import threading
from abc import ABCMeta, abstractmethod
from typing import Optional

from logzero import logger

from ..context.data_manager_context import DataManagerContext
from ..core.model import Model
from ..core.result_provider import ResultProvider


class SelectorStats(object):

    def __init__(self, dictionary):
        assert 'processed_objects' in dictionary, "Missing processed_objects attribute"
        self.dropped_objects = 0
        self.passed_objects = 0
        self.false_negatives = 0

        for key in dictionary:
            setattr(self, key, dictionary[key])


class Selector(metaclass=ABCMeta):

    @abstractmethod
    def add_result(self, result: ResultProvider) -> None:
        """Add processed results from model to selector"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clean up on end of mission"""
        pass

    @abstractmethod
    def add_context(self, context: DataManagerContext) -> None:
        """Add data manager context"""
        pass

    @abstractmethod
    def get_result(self) -> Optional[ResultProvider]:
        """Transmit selected results"""
        pass

    @abstractmethod
    def new_model(self, model: Optional[Model]) -> None:
        """Triggered when a new model is available"""
        pass

    @abstractmethod
    def get_stats(self) -> SelectorStats:
        """Returns current mission stats"""
        pass


class SelectorBase(Selector):

    def __init__(self):
        self.result_queue = queue.Queue(maxsize=100)
        self.stats_lock = threading.Lock()
        self.items_processed = 0
        self.num_revisited = 0
        self.num_positives = 0
        self.model_train_time = 0
        self.model_examples = 0

        self._model_lock = threading.Lock()
        self._model_present = False
        self._mission = None
        self.transmit_queue = None

        self._clear_event = threading.Event()

    @abstractmethod
    def _add_result(self, result: ResultProvider) -> None:
        """Helper function specific to selection strategy"""
        pass

    @abstractmethod
    def _new_model(self, model: Optional[Model]) -> None:
        """Helper function specific to selection strategy"""
        pass

    def add_context(self, context: DataManagerContext):
        self._mission = context

    def add_result(self, result: ResultProvider) -> None:
        """Add processed results from model to selection strategy"""
        with self._model_lock:
            model_present = self._model_present

        if not model_present:
            self.result_queue.put(result) # pass through
        else:
            self._add_result(result)

        with self.stats_lock:
            self.items_processed += 1
            '''if self.items_processed % 10 == 0:
                logger.info("Total items processed: {}".format(self.items_processed))'''

    def new_model(self, model: Optional[Model]) -> None:
        """New model generation is available from trainer""" 
        with self._model_lock:
            self._model_present = model is not None

        self._new_model(model)

    def clear(self) -> None:
        logger.info("Selector clear called in selector base...")

        self._clear_event.set()
        self.result_queue.put(None)


    def get_result(self) -> Optional[ResultProvider]:
        """Returns result for transmission when available"""
        while True:
            try:
                return self.result_queue.get(timeout=10)
            except queue.Empty:
                pass

    def get_stats(self) -> SelectorStats:
        with self.stats_lock:
            stats = {'processed_objects': self.items_processed,
                     'items_revisited': self.num_revisited,
                     'positive_in_stream': self.num_positives,
                     'train_positives': self.model_examples,
                     }

        return SelectorStats(stats)
