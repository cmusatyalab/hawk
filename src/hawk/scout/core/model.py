# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import errno
import os
import threading
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from logzero import logger
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)

from ...proto.messages_pb2 import ModelMetrics, TestResults
from ..context.model_trainer_context import ModelContext
from .object_provider import ObjectProvider
from .result_provider import ResultProvider
from .utils import log_exceptions


class Model(metaclass=ABCMeta):

    @abstractmethod
    def infer(self, requests: Iterable[ObjectProvider]) -> Iterable[ResultProvider]:
        pass

    @abstractmethod
    def load_model(self, path: Path) -> None:
        pass

    @abstractmethod
    def evaluate_model(self, test_path: Path) -> None:
        pass

    @abstractmethod
    def serialize(self) -> bytes:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @property
    @abstractmethod
    def version(self) -> int:
        pass

    @property
    @abstractmethod
    def mode(self) -> str:
        pass

    @property
    @abstractmethod
    def train_examples(self) -> Dict:
        pass

    @property
    @abstractmethod
    def train_time(self) -> int:
        pass

class ModelBase(Model):

    def __init__(self, 
                 args: Dict, 
                 model_path: Path,
                 context: ModelContext = None):

        self.context = context
        self.request_count = 0
        self.result_count = 0
        self._model_lock = threading.Lock()
        self._running = True
        
        if self.context is not None:
            self.request_queue = self.context.model_input_queue
            self.result_queue = self.context.model_output_queue        
        
        
        self._model = None
        self._version = args.get('version', 0)
        self._mode = args.get('mode', "hawk")
        self._train_examples = args.get('train_examples', {'1':0, '0':0})
        self._train_time = args.get('train_time', 0)

        if self._mode != "oracle" and not model_path.exists():
            raise FileNotFoundError(errno.ENOENT, 
                                    os.strerror(errno.ENOENT), 
                                    model_path)
    @property
    def version(self) -> int:
        return self._version

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def train_examples(self) -> Dict:
        return self._train_examples

    @property
    def train_time(self) -> int:
        return self._train_time

    @log_exceptions
    def preprocess(self, request):
        return request

    @log_exceptions
    def add_requests(self, request):
        if self.context is None:
            return
        self.request_count += 1
        self.request_queue.put(self.preprocess(request))
        if self.request_count == 1:
            threading.Thread(target=self._infer_results, name='model-infer').start()
        return

    @log_exceptions
    def get_results(self):
        if self.context is None:
            return
        return self.result_queue.get()

    @log_exceptions
    def _infer_results(self):
        while True:
            time.sleep(5)

    def is_running(self) -> bool:
        return self._running

    @staticmethod
    def calculate_performance(version: int, target: List[int], pred: List[float], is_probability: bool=True) \
            -> TestResults:
        assert len(target) == len(pred)
        pred = np.array(pred)
        target = np.array(target)

        ap = average_precision_score(target, pred, average=None)
        precision, recall, thresholds = precision_recall_curve(target, pred)
        f1_score = np.nan_to_num(2 * (precision * recall) / (precision + recall))
        f1_best_idx = np.argmax(f1_score)

        best_threshold = thresholds[f1_best_idx]
        logger.info('Test AUC: {}'.format(ap))

        pred = np.where(pred >= best_threshold, 1, 0) if is_probability else np.where(pred > 0, 1, 0)
        logger.info(
            'Test classification report ({} threshold):\n{}'.format(best_threshold, classification_report(target, pred)))

        stats = classification_report(target, pred, output_dict=True)
        
        tp = np.sum(target[target==1] == pred[target==1])
        fp = np.sum(target[target==1] == pred[target==0])
        fn = np.sum(target[target==0] == pred[target==1])

        logger.info("Total positive {} TP {} FP {} FN {}".format(tp, fp, fn))
        model_metrics = ModelMetrics(
            truePositives=tp,
            falsePositives=fp,
            falseNegatives=fn,
            precision=stats['1']['precision'],
            recall=stats['1']['recall'],
            f1Score=stats['1']['f1-score']
        )

        return TestResults(
            testExamples=len(pred),
            auc=ap,
            modelMetrics=model_metrics,
            bestThreshold=best_threshold.item(),
            # precisions=[item.item() for item in precision],
            # recalls=[item.item() for item in recall],
            version=version
        )

    def evaluate_model(self, test_path: Path) -> None:
        pass
    
            