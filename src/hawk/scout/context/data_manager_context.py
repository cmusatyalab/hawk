# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only


"""Abstract class for data manager context."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from .context_base import ContextBase

if TYPE_CHECKING:
    from pathlib import Path

    from ...classes import ClassCounter
    from ...proto.messages_pb2 import MissionId


class DataManagerContext(ContextBase):
    """Data Manager Context.

    Attributes
    ----------
    mission_id : MissionId
        unique id of mission
    data_dir : Path
        path to the TRAIN/TEST split

    """

    scml_deploy_options: dict[str, int]

    @property
    @abstractmethod
    def mission_id(self) -> MissionId:
        pass

    @property
    @abstractmethod
    def data_dir(self) -> Path:
        pass

    @abstractmethod
    def check_initial_model(self) -> bool:
        """Checks if initial model weights available in config phase."""

    @abstractmethod
    def new_labels_callback(
        self,
        sample_counts: ClassCounter,
        retrain: bool = True,
    ) -> None:
        """Adds new labels to the data directory."""

    @abstractmethod
    def log(self, msg: str, end_t: float | None = None) -> None:
        """When logging is enabled, logs 'msg' to the logfile."""

    @property
    @abstractmethod
    def model_version(self) -> int:
        """Returns the current model version used for inferencing."""
