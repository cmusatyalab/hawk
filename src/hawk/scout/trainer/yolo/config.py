# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from pathlib import Path

from ...core.config import ModelConfig, ModelTrainerConfig


class YOLOModelConfig(ModelConfig):
    input_size: int = 480
    test_batch_size: int = 32


class YOLOTrainerConfig(YOLOModelConfig, ModelTrainerConfig):
    train_batch_size: int = 16
    image_size: int = 640
    initial_model_epochs: int = 30
    test_dir: Path | None = None
