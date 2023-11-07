# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import queue
import threading
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

from logzero import logger

from ..context.data_manager_context import DataManagerContext
from ..core.model import Model
from ..core.result_provider import ResultProvider


@dataclass
class SelectorStats:
    surv_TPs: int
    surv_TNs: int
    surv_FPs: int
    surv_FNs: int
    surv_threat_not_neut: int
    num_countermeasures_remain: int
    processed_objects: int
    items_revisited: int
    positive_in_stream: int
    train_positives: int
    dropped_objects: int = 0
    passed_objects: int = 0
    false_negatives: int = 0


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
    def __init__(self) -> None:
        self.result_queue = queue.Queue(maxsize=100)
        self.stats_lock = threading.Lock()
        self.items_processed = 0
        self.num_revisited = 0
        self.num_positives = 0
        self.model_train_time = 0
        self.model_examples = 0
        self.countermeasure_threshold = 0.5  ## make this configurable from config file
        self.surv_TPs = 0
        self.surv_FPs = 0
        self.surv_FNs = 0
        self.surv_TNs = 0
        self.surv_threat_not_neut = 0  ## increment this number by 1 for every FN
        ## this will get incremeneted by 1 upon every TP when countermeasures = 0
        self.num_countermeasures = 50

        self._model_lock = threading.Lock()
        self._model_present = False
        self._mission: Optional[DataManagerContext] = None
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

    def add_context(self, context: DataManagerContext) -> None:
        self._mission = context

    def add_result(self, result: ResultProvider) -> None:
        """Add processed results from model to selection strategy"""
        with self._model_lock:
            model_present = self._model_present

        if not model_present:
            self.result_queue.put(result)  # pass through
        else:
            self._add_result(result)
            # logger.info(f"Label: {result.gt}, Score: {result.score}, ID: {result.id}")
            #logger.info("Countermeasure threshold: {}".format(self.countermeasure_threshold))
            #logger.info("Total countermeasures: {}".format(self.num_countermeasures))
            perceived_truth = result.score >= self.countermeasure_threshold
            if result.gt:
                if perceived_truth:
                    self.surv_TPs += 1
                    if self.num_countermeasures == 0:
                        self.surv_threat_not_neut += 1
                    else:
                        self.num_countermeasures -= 1
                else:
                    self.surv_FNs += 1

            else:
                if perceived_truth:
                    self.surv_FPs += 1
                    if self.num_countermeasures > 0:
                        self.num_countermeasures -= 1
                else:
                    self.surv_TNs += 1

            ## HEre is where well compare threshold to round truth and actual score to determine TP, TN, FP, FN

            ## Deploy countermeasure if sample is TP or FP

            ## Decrement number of countermeausures if TP or FP

        with self.stats_lock:
            self.items_processed += 1
            """if self.items_processed % 10 == 0:
                logger.info("Total items processed: {}".format(self.items_processed))"""

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
            stats = {
                "processed_objects": self.items_processed,
                "items_revisited": self.num_revisited,
                "positive_in_stream": self.num_positives,
                "train_positives": self.model_examples,
                "surv_TPs": self.surv_TPs,
                "surv_TNs": self.surv_TNs,
                "surv_FPs": self.surv_FPs,
                "surv_FNs": self.surv_FNs,
                "surv_threat_not_neut": self.surv_threat_not_neut,
                "num_countermeasures_remain": self.num_countermeasures,
            }

        return SelectorStats(**stats)
