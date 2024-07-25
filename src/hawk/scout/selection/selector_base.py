# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import queue
import threading
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from logzero import logger

from ..core.model import Model
from ..core.result_provider import ResultProvider
from ..stats import (
    HAWK_INFERENCED_OBJECTS,
    HAWK_SELECTOR_PRIORITY_QUEUE_LENGTH,
    HAWK_SELECTOR_RESULT_QUEUE_LENGTH,
    HAWK_SELECTOR_REVISITED_OBJECTS,
    HAWK_SELECTOR_SKIPPED_OBJECTS,
    HAWK_SURVIVABILITY_FALSE_NEGATIVES,
    HAWK_SURVIVABILITY_FALSE_POSITIVES,
    HAWK_SURVIVABILITY_THREATS_NOT_COUNTERED,
    HAWK_SURVIVABILITY_TRUE_NEGATIVES,
    HAWK_SURVIVABILITY_TRUE_POSITIVES,
    collect_metrics_total,
)

if TYPE_CHECKING:
    from ..core.mission import Mission


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


class Selector(metaclass=ABCMeta):
    @abstractmethod
    def add_result(self, result: ResultProvider | None) -> int:
        """Add processed results from model to selector"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clean up on end of mission"""
        pass

    @abstractmethod
    def add_context(self, context: Mission) -> None:
        """Add data manager context"""
        pass

    @abstractmethod
    def get_result(self) -> ResultProvider | None:
        """Transmit selected results"""
        pass

    @abstractmethod
    def new_model(self, model: Model | None) -> None:
        """Triggered when a new model is available"""
        pass

    @abstractmethod
    def get_stats(self) -> SelectorStats:
        """Returns current mission stats"""
        pass

    @abstractmethod
    def add_easy_negatives(self, path: Path) -> None:
        """Add unlabeled easy negatives to training set"""
        pass


class SelectorBase(Selector):
    def __init__(self, mission_id: str) -> None:
        self._mission: Mission | None = None

        self.result_queue: queue.Queue[ResultProvider | None] = queue.Queue(maxsize=100)

        self.items_processed = 0

        self.mission_id = mission_id
        self.inferenced_objects = HAWK_INFERENCED_OBJECTS

        # currently a no-op, we would need to change initialization order so that
        # the mission is created before selector, or at least pass the list of
        # classes to the SelectorBase initializer
        if self._mission is not None:
            # Hint to prometheus_client which labels we might use
            class_list = list(self._mission.class_manager.classes)
            for class_name in ["negative"] + class_list[1:]:
                HAWK_INFERENCED_OBJECTS.labels(
                    mission=mission_id, gt=class_name, model_version="0"
                )

        self.items_skipped = HAWK_SELECTOR_SKIPPED_OBJECTS.labels(mission=mission_id)
        self.priority_queue_length = HAWK_SELECTOR_PRIORITY_QUEUE_LENGTH.labels(
            mission=mission_id
        )
        self.result_queue_length = HAWK_SELECTOR_RESULT_QUEUE_LENGTH.labels(
            mission=mission_id
        )
        self.num_revisited = HAWK_SELECTOR_REVISITED_OBJECTS.labels(mission=mission_id)

        self.model_examples = 0

        self.countermeasure_threshold = 0.5  # make this configurable from config file
        self.num_countermeasures = 50
        self.surv_TPs = HAWK_SURVIVABILITY_TRUE_POSITIVES.labels(mission=mission_id)
        self.surv_FPs = HAWK_SURVIVABILITY_FALSE_POSITIVES.labels(mission=mission_id)
        self.surv_FNs = HAWK_SURVIVABILITY_FALSE_NEGATIVES.labels(mission=mission_id)
        self.surv_TNs = HAWK_SURVIVABILITY_TRUE_NEGATIVES.labels(mission=mission_id)

        # incremented by 1 for every FN and every TP when countermeasures remaining = 0
        self.surv_threat_not_neut = HAWK_SURVIVABILITY_THREATS_NOT_COUNTERED.labels(
            mission=mission_id
        )

        self._model_lock = threading.Lock()
        self._model_present = False
        self.transmit_queue = None

        self._clear_event = threading.Event()

    @abstractmethod
    def _add_result(self, result: ResultProvider) -> None:
        """Helper function specific to selection strategy"""
        pass

    @abstractmethod
    def _new_model(self, model: Model | None) -> None:
        """Helper function specific to selection strategy"""
        pass

    def add_context(self, context: Mission) -> None:
        self._mission = context

    def add_result(self, result: ResultProvider | None) -> int:
        """Add processed results from model to selection strategy"""
        if result is None:
            return self.items_processed

        self.items_processed += 1

        # collect inference stats
        # making sure the negative class is always called "negative" to make
        # graphing easier
        class_name = (
            "negative"
            if result.gt == 0
            else self._mission.class_manager.label_name_dict[result.gt]
            if self._mission is not None
            else str(result.gt)
        )
        model_version = str(result.model_version)
        self.inferenced_objects.labels(
            mission=self.mission_id,
            gt=class_name,
            model_version=model_version,
        ).observe(result.score)

        with self._model_lock:
            model_present = self._model_present

        if not model_present:
            self.items_skipped.inc()
            self.result_queue_length.inc()
            self.result_queue.put(result)  # pass through
        else:
            self._add_result(result)

            # logger.info(f"Label: {result.gt}, Score: {result.score}, ID: {result.id}")
            # logger.info(f"Countermeasure threshold: {self.countermeasure_threshold}")
            # logger.info(f"Total countermeasures: {self.num_countermeasures}")
            perceived_truth = result.score >= self.countermeasure_threshold
            if result.gt:
                if perceived_truth:
                    self.surv_TPs.inc()
                    countermeasures_used = collect_metrics_total(
                        self.surv_TPs
                    ) + collect_metrics_total(self.surv_FPs)
                    if countermeasures_used > self.num_countermeasures:
                        self.surv_threat_not_neut.inc()
                else:
                    self.surv_FNs.inc()
                    self.surv_threat_not_neut.inc()
            else:
                if perceived_truth:
                    self.surv_FPs.inc()
                    # self.countermeasures_used.inc()
                else:
                    self.surv_TNs.inc()

            # Here is where we'll compare threshold to ground truth and actual
            # score to determine TP, TN, FP, FN

            # Deploy countermeasure if sample is TP or FP

            # Decrement number of countermeausures if TP or FP

        # if self.items_processed % 10 == 0:
        #     logger.info(f"Total items processed: {self.items_processed}")
        return self.items_processed

    def new_model(self, model: Model | None) -> None:
        """New model generation is available from trainer"""
        with self._model_lock:
            self._model_present = model is not None

        self._new_model(model)

    def clear(self) -> None:
        logger.info("Selector clear called in selector base...")

        self._clear_event.set()
        self.result_queue_length.inc()
        self.result_queue.put(None)

    def get_result(self) -> ResultProvider | None:
        """Returns result for transmission when available"""
        while True:
            try:
                result = self.result_queue.get(timeout=10)
                self.result_queue_length.dec()
                return result
            except queue.Empty:
                pass

    def get_stats(self) -> SelectorStats:
        surv_TPs = collect_metrics_total(self.surv_TPs)
        surv_TNs = collect_metrics_total(self.surv_TNs)
        surv_FPs = collect_metrics_total(self.surv_FPs)
        surv_FNs = collect_metrics_total(self.surv_FNs)
        surv_threat_not_neut = collect_metrics_total(self.surv_threat_not_neut)

        countermeasures_used = surv_TPs + surv_FPs
        countermeasures_remain = max(0, self.num_countermeasures - countermeasures_used)

        inferenced = [
            sample
            for instance in self.inferenced_objects.collect()
            for sample in instance.samples
            if sample.name.endswith("_count")
        ]
        processed_objects = sum(sample.value for sample in inferenced)
        positives_in_stream = sum(
            sample.value for sample in inferenced if sample.labels["gt"] != "negative"
        )

        return SelectorStats(
            processed_objects=int(processed_objects),
            items_revisited=collect_metrics_total(self.num_revisited),
            positive_in_stream=int(positives_in_stream),
            train_positives=self.model_examples,
            surv_TPs=surv_TPs,
            surv_TNs=surv_TNs,
            surv_FPs=surv_FPs,
            surv_FNs=surv_FNs,
            surv_threat_not_neut=surv_threat_not_neut,
            num_countermeasures_remain=countermeasures_remain,
        )
