# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from pathlib import Path

import torch
from logzero import logger

from ...context.model_trainer_context import ModelContext
from ...core.model_trainer import ModelTrainer
from .config import FewShotTrainerConfig
from .model import FewShotModel

torch.multiprocessing.set_sharing_strategy("file_system")


class FewShotTrainer(ModelTrainer):
    config_class = FewShotTrainerConfig
    config: FewShotTrainerConfig

    def __init__(self, config: FewShotTrainerConfig, context: ModelContext):
        super().__init__(config, context)

        self._model_path: Path | None = None

        logger.info("FSL TRAINER CALLED")

    def load_model(
        self, path: Path | None = None, content: bytes = b"", version: int = -1
    ) -> FewShotModel:
        if path is None or not path.is_file():
            assert len(content)
            path = self.context.model_path(version)
            path.write_bytes(content)

        logger.info(f"Loading from path {path}")
        self._model_path = path
        return FewShotModel(self.config, self.context, path, version)

    def train_model(self, train_dir: Path) -> FewShotModel:
        version = self.get_new_version()
        return self.load_model(self._model_path, version=version)
