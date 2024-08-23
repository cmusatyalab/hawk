# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import threading
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from ..context.model_trainer_context import ModelContext
from .model import ModelBase


class ModelTrainer(metaclass=ABCMeta):
    @abstractmethod
    def load_model(
        self, path: Path | None = None, content: bytes = b"", version: int = -1
    ) -> ModelBase:
        pass

    @abstractmethod
    def train_model(self, train_dir: Path) -> ModelBase:
        pass


class ModelTrainerBase(ModelTrainer):
    def __init__(self, context: ModelContext, args: dict[str, str]):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self._latest_version = -1
        self._version_lock = threading.Lock()

        self.context = context

        self.args: dict[str, Any] = dict(mode="hawk")
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

    def capture_trainingset(self, cmd: str, extra_files: list[Path]) -> None:
        # assuming train dir is {mission_dir}/data/examples/train
        # and train.txt, val.txt and saved models are in {mission_dir}/model
        model_version = self.get_version()
        mission_dir = self.context.model_dir.parent
        archive = self.context.model_path(model_version, template="dataset-{}.zip")

        with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=1) as zf:
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
