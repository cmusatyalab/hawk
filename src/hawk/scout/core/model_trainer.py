# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import torch

from ...plugins import HawkPlugin
from .config import ModelMode, ModelTrainerConfig

if TYPE_CHECKING:
    from pathlib import Path

    from ..context.model_trainer_context import ModelContext
    from .model import ModelBase


class ModelTrainerBase(HawkPlugin, ABC):
    config_class = ModelTrainerConfig
    config: ModelTrainerConfig

    @abstractmethod
    def load_model(self, path: Path, version: int) -> ModelBase: ...

    @abstractmethod
    def train_model(self, train_dir: Path) -> ModelBase: ...


class ModelTrainer(ModelTrainerBase):
    def __init__(self, config: ModelTrainerConfig, context: ModelContext) -> None:
        super().__init__(config)
        self.context = context

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self._latest_version = -1
        self._version_lock = threading.Lock()

        self.prev_model_path: Path | None = None

    def get_new_version(self) -> int:
        with self._version_lock:
            self._latest_version += 1
            return self._latest_version

    def get_version(self) -> int:
        with self._version_lock:
            return self._latest_version

    def import_model(self, model: bytes) -> ModelBase:
        version = self.get_new_version()

        path = self.context.model_path(version)
        path.write_bytes(model)

        self.prev_model_path = path
        return self.load_model(path, version)

    def model_trainer(self, train_dir: Path) -> ModelBase:
        if self.config.mode == ModelMode.ORACLE and self.prev_model_path is not None:
            return self.load_model(self.prev_model_path, version=0)

        if self.config.mode == ModelMode.NOTIONAL:
            # sleep for training time
            time.sleep(self.config.notional_train_time)

            assert self.config.notional_model_path is not None
            return self.load_model(self.config.notional_model_path, version=0)

        return self.train_model(train_dir)

    def capture_trainingset(self, cmd: str, extra_files: list[Path]) -> None:
        if not self.config.capture_trainingset:
            return

        # assuming train dir is {mission_dir}/data/examples/train
        # and train.txt, val.txt and saved models are in {mission_dir}/model
        model_version = self.get_version()
        mission_dir = self.context.model_dir.parent
        archive = self.context.model_path(model_version, template="dataset-{}.zip")

        compression = (
            ZIP_DEFLATED
            if self.config.capture_trainingset_compresslevel
            else ZIP_STORED
        )
        level = self.config.capture_trainingset_compresslevel

        with ZipFile(archive, "w", compression=compression, compresslevel=level) as zf:
            zf.writestr("train.sh", cmd)
            for path in extra_files:
                if path.is_file():
                    arcname = path.relative_to(mission_dir.parent)
                    zf.write(path, arcname)
                    continue

                for file in path.rglob("*"):
                    if file.is_file():
                        arcname = file.relative_to(mission_dir.parent)
                        zf.write(file, arcname)
