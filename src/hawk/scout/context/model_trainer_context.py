# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only


"""Abstract class for model context"""

import multiprocessing as mp
from abc import abstractmethod
from pathlib import Path
from typing import Any, Tuple

from ..core.object_provider import ObjectProvider
from ..core.result_provider import ResultProvider
from .context_base import ContextBase


class ModelContext(ContextBase):
    novel_class_discovery: bool
    sub_class_discovery: bool

    def __init__(self) -> None:
        super().__init__()
        self.model_input_queue: mp.Queue[Tuple[ObjectProvider, Any]] = mp.Queue()
        self.model_output_queue: mp.Queue[ResultProvider] = mp.Queue()

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

    @abstractmethod
    def check_create_test(self) -> bool:
        """Checks if TEST dataset needs to be created"""
        pass

    def model_path(self, version: int, template: str = "model-{}.pth") -> Path:
        return self.model_dir / template.format(str(version).zfill(3))
