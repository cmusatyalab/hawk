# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

from abc import abstractmethod
from pathlib import Path
from typing import List

from hawk.context.context_base import ContextBase
from hawk.core.model_trainer import ModelTrainer
from hawk.proto.messages_pb2 import MissionId


class DataManagerContext(ContextBase):

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
        pass

    @abstractmethod
    def new_examples_callback(self, new_positives: int, 
                              new_negatives: int, retrain: bool = True) -> None:
        pass


