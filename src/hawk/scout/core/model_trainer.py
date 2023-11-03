# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import threading
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .model import ModelBase


class ModelTrainer(metaclass=ABCMeta):
    @abstractmethod
    def load_model(
        self, path: Optional[Path] = None, content: bytes = b"", version: int = -1
    ) -> ModelBase:
        pass

    @abstractmethod
    def train_model(self, train_dir: Path) -> ModelBase:
        pass


class ModelTrainerBase(ModelTrainer):
    def __init__(self, args: Dict[str, str]):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self._latest_version = -1
        self._version_lock = threading.Lock()

        self.args: Dict[str, Any] = dict(mode="hawk")
        self.args.update(args)

    def parse_args(self) -> None:
        raise NotImplementedError("Parse Args")

    def get_new_version(self) -> int:
        with self._version_lock:
            self._latest_version += 1
            version = self._latest_version
        return version

    def get_version(self) -> int:
        with self._version_lock:
            version = self._latest_version
        return version

    def set_version(self, version: int) -> int:
        with self._version_lock:
            self._latest_version = version
        return version
