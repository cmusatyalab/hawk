# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only


"""Abstract class for data manager context
"""

from abc import abstractmethod
from pathlib import Path
from typing import List

from hawk.context.context_base import ContextBase
from hawk.core.model_trainer import ModelTrainer
from hawk.proto.messages_pb2 import MissionId


class DataManagerContext(ContextBase):
    """Data Manager Context

    Attributes
    ----------
    mission_id : MissionId
        unique id of mission
    data_dir : Path
        path to the TRAIN/TEST split
    """
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
        """Checks if intital model weights available in config phase"""
        pass

    @abstractmethod
    def new_labels_callback(self, new_positives: int, 
                              new_negatives: int, retrain: bool = True) -> None:
        """Adds new labels to the data directory"""
        pass

    @abstractmethod
    def check_create_test(self) -> bool:
        """Checks if TEST dataset needs to be created"""
        pass

