# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from ...core.config import ModelConfig, ModelTrainerConfig


class ActivityModelConfig(ModelConfig):
    embed_dim: int = 480
    test_batch_size: int = 1


class ActivityTrainerConfig(ActivityModelConfig, ModelTrainerConfig):
    T: int = 5
    depth: int = 2
    num_heads: int = 16
    mlp_dim: int = 1920
    num_classes: int = 2
    head_dim: int = 480
    train_batch_size: int = 1
    initial_model_epochs: int = 10
    online_epochs: int | list[tuple[int, int]] = 10

    notional_train_time: float = 0.0
