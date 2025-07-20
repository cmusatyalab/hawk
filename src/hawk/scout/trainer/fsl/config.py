# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from pathlib import Path

from ...core.config import ModelConfig, ModelTrainerConfig


class FSLModelConfig(ModelConfig):
    arch: str = "siamese"
    input_size: int = 256
    test_batch_size: int = 64
    support_path: Path = Path("/srv/diamond/dota/support.jpg")


class FSLTrainerConfig(FSLModelConfig, ModelTrainerConfig):
    support_data: str | None  # base64-encoded jpeg image loaded from example_path
    fsl_traindir: Path = Path("/srv/diamond/dota/fsl_traindir")
