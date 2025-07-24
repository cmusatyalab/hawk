# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from logzero import logger

from ...core.model_trainer import ModelTrainer
from .config import FewShotTrainerConfig
from .model import FewShotModel

if TYPE_CHECKING:
    from pathlib import Path

    from ...context.model_trainer_context import ModelContext

torch.multiprocessing.set_sharing_strategy("file_system")


class FewShotTrainer(ModelTrainer):
    config_class = FewShotTrainerConfig
    config: FewShotTrainerConfig

    def __init__(self, config: FewShotTrainerConfig, context: ModelContext) -> None:
        super().__init__(config, context)
        logger.info("FSL TRAINER CALLED")

    def load_model(self, path: Path, version: int) -> FewShotModel:
        logger.info(f"Loading from path {path}")
        return FewShotModel(self.config, self.context, path, version)

    def train_model(self, train_dir: Path) -> FewShotModel:
        assert self.prev_model_path is not None
        return self.load_model(self.prev_model_path, 0)
