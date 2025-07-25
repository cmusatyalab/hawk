# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import queue

    from ..core.model import Model
    from ..core.result_provider import ResultProvider
    from ..retrieval.retriever import Retriever

    ReexaminationQueueType = queue.PriorityQueue[Tuple[float, float, ResultProvider]]


class ReexaminationStrategy(metaclass=ABCMeta):
    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    @property
    @abstractmethod
    def reexamines_old_results(self) -> bool:
        """Returns True if old results are reexamined by strategy."""

    @abstractmethod
    def get_new_queues(
        self,
        model: Model,
        old_queues: ReexaminationQueueType,
        start_time: float = 0,
    ) -> tuple[ReexaminationQueueType, int]:
        """Generates a new queue with reexamined results."""
