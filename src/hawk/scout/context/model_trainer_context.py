# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only


"""Abstract class for model context."""

from __future__ import annotations

import multiprocessing as mp
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from .context_base import ContextBase

if TYPE_CHECKING:
    from pathlib import Path

    from ...objectid import ObjectId
    from ..core.result_provider import ResultProvider
    from ..retrieval.retriever import Retriever


class ModelContext(ContextBase):
    novel_class_discovery: bool
    sub_class_discovery: bool

    def __init__(self, retriever: Retriever) -> None:
        super().__init__()
        self.model_input_queue: mp.Queue[tuple[ObjectId, Any]] = mp.Queue()
        self.model_output_queue: mp.Queue[ResultProvider] = mp.Queue()
        self.retriever = retriever

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
        """Checks if TEST dataset needs to be created."""

    def model_path(self, version: int, template: str = "model-{}.pth") -> Path:
        return self.model_dir / template.format(str(version).zfill(3))
