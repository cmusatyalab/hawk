# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

from abc import ABCMeta, abstractmethod
import errno
import os
import threading
import time
from pathlib import Path
from typing import Iterable, Dict

from hawk.context.model_trainer_context import ModelTrainerContext
from hawk.core.object_provider import ObjectProvider
from hawk.core.result_provider import ResultProvider
from hawk.core.utils import log_exceptions

class Model(metaclass=ABCMeta):

    @abstractmethod
    def infer(self, requests: Iterable[ObjectProvider]) -> Iterable[ResultProvider]:
        pass

    @abstractmethod
    def load_model(self, path: Path) -> None:
        pass

    @abstractmethod
    def evaluate_model(self, directory: Path) -> None:
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
    def train_examples(self) -> int:
        pass

class ModelBase(Model):

    def __init__(self, 
                 args: Dict, 
                 model_path: Path,
                 context: ModelTrainerContext = None):

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
        self._train_examples = {'0':0, '1':0}

        if self._mode != "oracle" and not model_path.exists():
            raise FileNotFoundError(errno.ENOENT, 
                                    os.strerror(errno.ENOENT), 
                                    model_path)
    @property
    def version(self) -> int:
        return self._version

    @property
    def train_examples(self) -> int:
        return self._train_examples

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
    
            