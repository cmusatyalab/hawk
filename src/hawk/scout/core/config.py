# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic_settings import SettingsConfigDict

from ...plugins import HawkPluginConfig


class ModelMode(Enum):
    HAWK = "hawk"
    ORACLE = "oracle"
    NOTIONAL = "notional"


class ModelConfig(HawkPluginConfig):
    model_config = SettingsConfigDict(
        env_prefix="hawk_model_",
        env_file="hawk.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mode: ModelMode = ModelMode.HAWK


class ModelTrainerConfig(ModelConfig):
    pass


class ModelHomeConfig(ModelTrainerConfig):
    initial_model_path: Optional[Path] = None
    base_model_path: Optional[Path] = None
    bootstrap_path: Optional[Path] = None
    train_validate: bool = True
