# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

from abc import ABCMeta, abstractmethod
from collections import Sized
from typing import Iterable

from hawk.core.object_provider import ObjectProvider
from hawk.proto.messages_pb2 import HawkObject
from hawk.retrieval.retriever_stats import RetrieverStats
from hawk.context.data_manager_context import DataManagerContext


class Retriever(metaclass=ABCMeta):

    def __init__(self) -> None:
        self._context = None

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def get_objects(self) -> Iterable[ObjectProvider]:
        pass

    @abstractmethod
    def get_object(self, object_id: str, attributes: Sized) -> HawkObject:
        pass

    @abstractmethod
    def get_stats(self) -> RetrieverStats:
        pass

    @abstractmethod
    def is_running(self) -> RetrieverStats:
        pass

    def add_context(self, context: DataManagerContext):
        self._context = context

