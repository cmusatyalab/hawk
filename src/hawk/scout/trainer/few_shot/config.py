# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from pathlib import Path

from ...core.config import ModelConfig, ModelTrainerConfig


class FewShotModelConfig(ModelConfig):
    support_data: bytes
    test_batch_size: int


class FewShotTrainerConfig(FewShotModelConfig, ModelTrainerConfig):
    train_batch_size: int = 64
    unfreeze_layers: int = 3
    initial_model_epochs: int = 15
    online_epochs: int | list[tuple[int, int]] = [(10, 0), (15, 100)]
    test_dir: Path | None = None

    notional_train_time: float = 0.0
