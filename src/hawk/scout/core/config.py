# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import model_validator
from pydantic_settings import SettingsConfigDict
from typing_extensions import Self

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
    # set compress level to 0 for no compression
    capture_trainingset: bool = True
    capture_trainingset_compresslevel: int = 1

    # for mode == NOTIONAL
    notional_model_path: Optional[Path] = None
    notional_train_time: float = 0.0

    @model_validator(mode="after")
    def notional_path_exists(self) -> Self:
        if self.mode == ModelMode.NOTIONAL and (
            self.notional_model_path is None or not self.notional_model_path.is_file()
        ):
            msg = f"Notional model path {self.notional_model_path} does not exist"
            raise ValueError(msg)
        return self


class ModelHomeConfig(ModelTrainerConfig):
    initial_model_path: Optional[Path] = None
    base_model_path: Optional[Path] = None
    bootstrap_path: Optional[Path] = None
    train_validate: bool = True
