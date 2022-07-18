# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

from abc import abstractmethod
from pathlib import Path
import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from hawk.context.context_base import ContextBase


class ModelTrainerContext(ContextBase):

    def __init__(self):
        self.result_queue = mp.Queue()
        self.model_input_queue = mp.Queue()
        self.model_output_queue = mp.Queue() 

    @property
    @abstractmethod
    def port(self) -> int:
        pass

    @property
    @abstractmethod
    def tb_writer(self) -> SummaryWriter:
        pass

    @property
    @abstractmethod
    def model_dir(self) -> Path:
        pass

    @abstractmethod
    def stop_model(self) -> None:
        pass
