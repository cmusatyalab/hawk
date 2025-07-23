# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import torch

from ...plugins import HawkPlugin
from ..context.model_trainer_context import ModelContext
from .config import ModelTrainerConfig
from .model import ModelBase


class ModelTrainerBase(HawkPlugin, ABC):
    config_class = ModelTrainerConfig
    config: ModelTrainerConfig

    @abstractmethod
    def load_model(
        self, path: Path | None = None, content: bytes = b"", version: int = -1
    ) -> ModelBase:
        pass

    @abstractmethod
    def train_model(self, train_dir: Path) -> ModelBase:
        pass


class ModelTrainer(ModelTrainerBase):
    def __init__(self, config: ModelTrainerConfig, context: ModelContext):
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
