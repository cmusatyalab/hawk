# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Top-k selection with one priority queue per class.

The topk selector ranks every result by the sum of its positive class scores,
so a rare class only gets picked when it outscores the common ones. Here every
result is queued once per positive class, ranked by that class's score, and a
scheduler decides which queue each of the k picks in a batch comes from. The
labeling budget is the same as topk: k results for every batch_size inferred.

Schedulers:
    round_robin  take turns between the classes
    deficit      pick from the class with the fewest labeled examples so far

With warm_start N the picks are made as topk would make them until the model
reaches version N. A new model ranks each class poorly, and splitting the
picks between the classes finds fewer positives to train the next model on.
"""

from __future__ import annotations

import copy
import queue
from collections import Counter
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

from logzero import logger

from .topk_selector import TopKSelector

if TYPE_CHECKING:
    from ...classes import ClassName
    from ...objectid import ObjectId
    from ..core.model import Model
    from ..core.result_provider import ResultProvider
    from ..reexamination.reexamination_strategy import (
        ReexaminationQueueType,
        ReexaminationStrategy,
    )

SCHEDULERS = ("round_robin", "deficit")


def class_score(result: ResultProvider, class_name: ClassName) -> float:
    for detection in result.bboxes:
        if detection.class_name == class_name:
            return detection.confidence
    return 0.0


class ClassScoreModel:
    """Lets a reexamination strategy rank results by a single class.

    Every object is inferred once by the new model and the result is shared
    between the classes, each class sees a copy scored by that class alone.
    """

    def __init__(self, model: Model) -> None:
        self.model = model
        self.class_name: ClassName | None = None
        self.rescored: dict[ObjectId, ResultProvider] = {}

    def infer(self, requests: Sequence[ObjectId]) -> Iterator[ResultProvider]:
        assert self.class_name is not None
        missing = [
            object_id for object_id in requests if object_id not in self.rescored
        ]
        if missing:
            for result in self.model.infer(missing):
                self.rescored[result.id] = result

        for object_id in requests:
            if object_id not in self.rescored:
                continue
            result = copy.copy(self.rescored[object_id])
            result.score = class_score(result, self.class_name)
            yield result


class ClassAwareSelector(TopKSelector):
    def __init__(
        self,
        mission_id: str,
        k: int,
        batch_size: int,
        scheduler: str,
        countermeasure_threshold: float,
        total_countermeasures: int,
        reexamination_strategy: ReexaminationStrategy,
        add_negatives: bool = True,
        warm_start: int = 0,
    ) -> None:
        if scheduler not in SCHEDULERS:
            msg = (
                f"Unknown class-aware scheduler {scheduler!r}, use one of {SCHEDULERS}"
            )
            raise ValueError(msg)

        super().__init__(
            mission_id,
            k,
            batch_size,
            countermeasure_threshold,
            total_countermeasures,
            reexamination_strategy,
            add_negatives,
        )
        logger.info(f"Class-aware scheduler: {scheduler}, warm start {warm_start}")
        self._scheduler = scheduler
        self._class_queues: dict[ClassName, ReexaminationQueueType] = {}

        # ranked by the sum of the positive class scores, as in topk; only
        # kept until the model reaches the warm start version
        self._warm_start = warm_start
        self._topk_queue: ReexaminationQueueType | None = (
            queue.PriorityQueue() if warm_start > 0 else None
        )

        # a sent result stays in the queues of the other classes until it
        # reaches the front, where it is dropped
        self._sent: set[ObjectId] = set()
        self._num_queued = 0

        self._next_class = 0  # round_robin position
        self._picks_this_batch: Counter[ClassName] = Counter()  # deficit

    def _classes(self) -> list[ClassName]:
        assert self._mission is not None
        return list(self._mission.class_list.positive)

    def _enqueue(self, result: ResultProvider, time_result: float) -> None:
        for class_name in self._classes():
            class_queue = self._class_queues.setdefault(
                class_name, queue.PriorityQueue()
            )
            class_queue.put((-class_score(result, class_name), time_result, result))
        if self._topk_queue is not None:
            self._topk_queue.put((-result.score, time_result, result))
        self._num_queued += 1

    def _peek(self, class_name: ClassName) -> ResultProvider | None:
        class_queue = self._class_queues.get(class_name)
        while class_queue is not None and not class_queue.empty():
            result = class_queue.queue[0][-1]
            if result.id not in self._sent:
                return result
            class_queue.get_nowait()
        return None

    def _pick_class(self, candidates: list[ClassName]) -> ClassName:
        if self._scheduler == "round_robin":
            return self._round_robin(candidates)
        return self._fewest_labeled(candidates)

    def _round_robin(self, candidates: list[ClassName]) -> ClassName:
        classes = self._classes()
        for offset in range(len(classes)):
            index = (self._next_class + offset) % len(classes)
            if classes[index] in candidates:
                self._next_class = (index + 1) % len(classes)
                return classes[index]
        return candidates[0]

    def _fewest_labeled(self, candidates: list[ClassName]) -> ClassName:
        # count picks of this batch as if their labels were already in, so the
        # k picks of one batch don't all go to the same class. ties go to the
        # earlier class in the class list.
        assert self._mission is not None
        labeled = self._mission._data_manager.class_counts.counter
        return min(candidates, key=lambda c: labeled[c] + self._picks_this_batch[c])

    def select_tiles(self, num_tiles: int) -> None:
        self._picks_this_batch.clear()
        super().select_tiles(num_tiles)

    def _warm(self) -> bool:
        if self._topk_queue is not None and self.version >= self._warm_start:
            logger.info(f"Class-aware warm start done at model version {self.version}")
            self._topk_queue = None
        return self._topk_queue is not None

    def _next_topk_result(self) -> ResultProvider | None:
        assert self._topk_queue is not None
        while not self._topk_queue.empty():
            result = self._topk_queue.get_nowait()[-1]
            if result.id not in self._sent:
                self._sent.add(result.id)
                self._num_queued -= 1
                logger.info(f"Class-aware warm start pick: {result.id}")
                return result
        return None

    def _next_result(self) -> ResultProvider | None:
        if self._warm():
            return self._next_topk_result()

        candidates = [c for c in self._classes() if self._peek(c) is not None]
        if not candidates:
            return None

        class_name = self._pick_class(candidates)
        result = self._class_queues[class_name].get_nowait()[-1]
        self._sent.add(result.id)
        self._num_queued -= 1
        self._picks_this_batch[class_name] += 1
        logger.info(
            f"Class-aware pick from {class_name} queue: {result.id} "
            f"{class_name} score {class_score(result, class_name):.4f}",
        )
        return result

    def _unsent(self) -> Iterable[ResultProvider]:
        seen: set[ObjectId] = set(self._sent)
        for class_queue in self._class_queues.values():
            for *_, result in class_queue.queue:
                if result.id not in seen:
                    seen.add(result.id)
                    yield result

    def _queued_results(self) -> list[ResultProvider]:
        with self._insert_lock:
            return list(self._unsent())

    def _reexamine(self, model: Model) -> int:
        assert self._mission is not None
        scorer = ClassScoreModel(model)
        num_revisited = 0

        for class_name, class_queue in self._class_queues.items():
            unsent: ReexaminationQueueType = queue.PriorityQueue()
            for item in class_queue.queue:
                if item[-1].id not in self._sent:
                    unsent.put(item)

            scorer.class_name = class_name
            new_queue, count = self._reexamination_strategy.get_new_queues(
                scorer,  # type: ignore[arg-type]
                unsent,
                self._mission.start_time,
            )
            num_revisited = max(num_revisited, count)

            # queue the full rescored result, ranked by this class's score
            restored: ReexaminationQueueType = queue.PriorityQueue()
            for priority, time_result, result in new_queue.queue:
                restored.put(
                    (priority, time_result, scorer.rescored.get(result.id, result))
                )
            self._class_queues[class_name] = restored

        if self._warm():
            # rank by the new model's score where it rescored a result
            assert self._topk_queue is not None
            topk_queue: ReexaminationQueueType = queue.PriorityQueue()
            for _, time_result, result in self._topk_queue.queue:
                if result.id not in self._sent:
                    result = scorer.rescored.get(result.id, result)
                    topk_queue.put((-result.score, time_result, result))
            self._topk_queue = topk_queue

        # sent results were left out of every rebuilt queue
        self._sent.clear()
        self._num_queued = sum(1 for _ in self._unsent())
        self.priority_queue_length.set(self._num_queued)

        # the caller counts this towards the next batch, report what a single
        # queue got (as topk would) so the labeling budget stays the same
        return num_revisited

    def _reset_queues(self) -> None:
        self._class_queues = {}
        if self._topk_queue is not None:
            self._topk_queue = queue.PriorityQueue()
        self._sent.clear()
        self._num_queued = 0
        self.priority_queue_length.set(0)
