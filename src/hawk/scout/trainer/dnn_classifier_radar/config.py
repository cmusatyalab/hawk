# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from ...core.config import ModelConfig, ModelTrainerConfig


class DNNRadarModelConfig(ModelConfig):
    arch: str = "resnet50"
    input_size: int = 224
    test_batch_size: int = 64
    pick_patches: bool = False


class DNNRadarTrainerConfig(DNNRadarModelConfig, ModelTrainerConfig):
    train_batch_size: int = 64
    unfreeze_layers: int = 3
    initial_model_epochs: int = 15
    online_epochs: int | list[tuple[int, int]] = [(10, 0), (15, 100)]

    notional_train_time: float = 0.0
