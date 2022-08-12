# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import threading
import torch

from abc import ABCMeta, abstractmethod
from enum import Enum, auto, unique
from pathlib import Path
from typing import Dict
from google.protobuf.any_pb2 import Any

from hawk.core.model import Model

class ModelTrainer(metaclass=ABCMeta):

    @abstractmethod
    def load_model(self, path: str, content: bytes, version: int) -> Model:
        pass

    @abstractmethod
    def train_model(self, train_dir: Path) -> Model:
        pass
    

class ModelTrainerBase(ModelTrainer):

    def __init__(self, args: Dict[str, str]):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self._latest_version = -1
        self._version_lock = threading.Lock()
        self.args = dict(args)
        self.args['mode'] = self.args.get('mode', "hawk")

    def parse_args(self):
        # mode = args.get('mode', 'hawk') 
        # # Only applicable for baseline and oracle mode
        # wait_time = args.get('time', 0) 
        # model_dir = args.get('model_dir', "") 
        raise NotImplementedError("Parse Args") 
        pass    
    
    def get_new_version(self):
        with self._version_lock:
            self._latest_version += 1
            version = self._latest_version
        return version

    def get_version(self):
        with self._version_lock:
            version = self._latest_version
        return version

    def set_version(self, version):
        with self._version_lock:
            self._latest_version = version
        return version

