# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from pathlib import Path
from typing import Dict, Optional

import torch
from logzero import logger

from ...context.model_trainer_context import ModelContext
from ...core.model_trainer import ModelTrainerBase
from .model import FewShotModel

torch.multiprocessing.set_sharing_strategy("file_system")


class FewShotTrainer(ModelTrainerBase):
    def __init__(self, context: ModelContext, args: Dict[str, str]):
        assert "support_data" in args
        super().__init__(args)

        self.context = context
        self._model_path = None
        self.version = 0

        logger.info("FSL TRAINER CALLED")

    def load_model(
        self, path: Optional[Path] = None, content: bytes = b"", version: int = -1
    ) -> FewShotModel:
        if path is None or not path.is_file():
            assert len(content)
            path = self.context.model_path(version)
            path.write_bytes(content)

        logger.info(f"Loading from path {path}")
        self._model_path = path
        return FewShotModel(self.args, path, version, context=self.context)

    def train_model(self, train_dir: Path) -> FewShotModel:
        return self.load_model(self._model_path, version=self.version)
