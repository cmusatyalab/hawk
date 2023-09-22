# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only


"""Abstract class for model context
"""

import multiprocessing as mp
from abc import abstractmethod
from pathlib import Path

from .context_base import ContextBase


class ModelContext(ContextBase):
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
    def model_dir(self) -> Path:
        pass

    @abstractmethod
    def stop_model(self) -> None:
        pass

    def model_path(self, version: int, template: str = "model-{}.pth") -> Path:
        return self.model_dir / template.format(str(version).zfill(3))
