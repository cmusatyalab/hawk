# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

from abc import ABCMeta, abstractmethod
from typing import Optional
from pathlib import Path

from hawk.context.data_manager_context import DataManagerContext
from hawk.core.model import Model
from hawk.core.result_provider import ResultProvider
from hawk.selection.selector_stats import SelectorStats


class Selector(metaclass=ABCMeta):

    @abstractmethod
    def add_result(self, result: ResultProvider) -> None:
        pass

    @abstractmethod
    def finish(self) -> None:
        pass

    @abstractmethod
    def add_context(self, context: DataManagerContext) -> None:
        pass

    @abstractmethod
    def get_result(self) -> Optional[ResultProvider]:
        pass

    @abstractmethod
    def new_model(self, model: Optional[Model]) -> None:
        pass

    @abstractmethod
    def add_easy_negatives(self, path: Path) -> None:
        pass

    @abstractmethod
    def get_stats(self) -> SelectorStats:
        pass
