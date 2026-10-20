# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Reexamination must rank results by the score of the *new* model."""

from __future__ import annotations

import queue
import random
from typing import ClassVar, Iterable, Sequence

import pytest

from hawk import ObjectId
from hawk.scout.core.result_provider import ResultProvider
from hawk.scout.reexamination.full_reexamination_strategy import (
    FullReexaminationStrategy,
)
from hawk.scout.reexamination.no_reexamination_strategy import (
    NoReexaminationStrategy,
)
from hawk.scout.reexamination.top_reexamination_strategy import (
    TopReexaminationStrategy,
)

pytestmark = pytest.mark.scout


def make_result(name: str, score: float) -> ResultProvider:
    object_id = ObjectId(f"/negative/collection/id/{name}")
    return ResultProvider(object_id, score, [], model_version=0)


def name_of(object_id: ObjectId) -> str:
    return object_id.oid.rsplit("/", 1)[-1]


class FakeModel:
    """Stands in for a newly trained model, re-scores objects from a table."""

    def __init__(self, new_scores: dict[str, float]) -> None:
        self.new_scores = new_scores

    def infer(self, requests: Sequence[ObjectId]) -> Iterable[ResultProvider]:
        for object_id in requests:
            name = name_of(object_id)
            yield make_result(name, self.new_scores[name])


# scores assigned by the old model; the new model reverses the ranking
OLD_SCORES = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5, "f": 0.4}
NEW_SCORES = {"a": 0.05, "b": 0.1, "c": 0.2, "d": 0.3, "e": 0.6, "f": 0.99}


def make_queue(
    scores: dict[str, float],
) -> queue.PriorityQueue[tuple[float, float, ResultProvider]]:
    pq: queue.PriorityQueue[tuple[float, float, ResultProvider]] = queue.PriorityQueue()
    for position, (name, score) in enumerate(scores.items()):
        pq.put((-score, float(position), make_result(name, score)))
    return pq


def drain(
    pq: queue.PriorityQueue[tuple[float, float, ResultProvider]],
) -> list[tuple[float, ResultProvider]]:
    items = []
    while not pq.empty():
        priority, _, result = pq.get_nowait()
        items.append((priority, result))
    return items


def names(items: list[tuple[float, ResultProvider]]) -> list[str]:
    return [name_of(result.id) for _, result in items]


@pytest.mark.parametrize(
    "strategy",
    [FullReexaminationStrategy(None), TopReexaminationStrategy(None, 3)],  # type: ignore[arg-type]
    ids=["full", "top"],
)
def test_priority_is_the_new_score(strategy):
    new_queue, count = strategy.get_new_queues(
        FakeModel(NEW_SCORES),
        make_queue(OLD_SCORES),
    )

    items = drain(new_queue)
    assert count == len(items)
    for priority, result in items:
        name = name_of(result.id)
        assert priority == pytest.approx(-NEW_SCORES[name])
        assert result.score == pytest.approx(NEW_SCORES[name])


def test_full_reranks_by_new_score():
    strategy = FullReexaminationStrategy(None)  # type: ignore[arg-type]
    new_queue, count = strategy.get_new_queues(
        FakeModel(NEW_SCORES),
        make_queue(OLD_SCORES),
    )

    assert count == len(OLD_SCORES)
    assert names(drain(new_queue)) == ["f", "e", "d", "c", "b", "a"]


def test_top_only_reexamines_the_top_k():
    strategy = TopReexaminationStrategy(None, 3)  # type: ignore[arg-type]
    new_queue, count = strategy.get_new_queues(
        FakeModel(NEW_SCORES),
        make_queue(OLD_SCORES),
    )

    # only the three best results of the old model (a, b, c) are re-scored
    assert count == 3
    assert names(drain(new_queue)) == ["c", "b", "a"]


@pytest.mark.parametrize(
    ("strategy", "reexamined"),
    [
        (FullReexaminationStrategy(None), ["a", "b", "c", "d", "e", "f"]),  # type: ignore[arg-type]
        (TopReexaminationStrategy(None, 3), ["a", "b", "c"]),  # type: ignore[arg-type]
    ],
    ids=["full", "top"],
)
def test_reexamined_results_compete_with_new_results(strategy, reexamined):
    """New results are queued with priority -score, so reexamined results must
    be interleaved with them according to their new score.
    """
    new_queue, _ = strategy.get_new_queues(
        FakeModel(NEW_SCORES),
        make_queue(OLD_SCORES),
    )
    new_queue.put((-0.5, 1e9, make_result("fresh", 0.5)))

    scores = {name: NEW_SCORES[name] for name in reexamined}
    scores["fresh"] = 0.5
    expected = sorted(scores, key=lambda name: -scores[name])
    assert names(drain(new_queue)) == expected


def test_top_with_empty_queue_is_a_noop():
    strategy = TopReexaminationStrategy(None, 3)  # type: ignore[arg-type]
    empty: queue.PriorityQueue[tuple[float, float, ResultProvider]] = (
        queue.PriorityQueue()
    )

    new_queue, count = strategy.get_new_queues(FakeModel({}), empty)

    assert count == 0
    assert new_queue is empty


def test_no_reexamination_discards_the_queue():
    strategy = NoReexaminationStrategy(None)  # type: ignore[arg-type]

    new_queue, count = strategy.get_new_queues(
        FakeModel(NEW_SCORES),
        make_queue(OLD_SCORES),
    )

    assert count == 0
    assert new_queue.empty()


class StubMission:
    start_time = 0.0

    def mission_time(self) -> float:
        return 0.0

    def log(self, msg: str, end_t: float | None = None) -> None:
        pass


class VersionedFakeModel(FakeModel):
    train_examples: ClassVar[dict[str, int]] = {"0": 0, "1": 0}

    def __init__(self, new_scores: dict[str, float], version: int) -> None:
        super().__init__(new_scores)
        self.version = version

    def is_oracle(self) -> bool:
        return False


@pytest.mark.parametrize(
    "strategy",
    [FullReexaminationStrategy(None), TopReexaminationStrategy(None, 3)],  # type: ignore[arg-type]
    ids=["full", "top"],
)
def test_selector_sends_best_rescored_result_first(strategy):
    """After a model update the selector must upload the result the new model
    likes best, not the one the previous model liked best.
    """
    from hawk.scout.selection.topk_selector import TopKSelector

    selector = TopKSelector("test-mission", 1, 100, 0.5, 50, strategy)
    selector.add_context(StubMission())  # type: ignore[arg-type]

    selector.new_model(VersionedFakeModel(OLD_SCORES, version=0))  # type: ignore[arg-type]
    for name, score in OLD_SCORES.items():
        selector.add_result(make_result(name, score))

    selector.new_model(VersionedFakeModel(NEW_SCORES, version=1))  # type: ignore[arg-type]
    selector.select_tiles(1)

    first = selector.result_queue.get_nowait()
    # the old model preferred "a", the new model scores "c" highest of the
    # top three results (and "f" highest overall when everything is rescored).
    expected = "f" if isinstance(strategy, FullReexaminationStrategy) else "c"
    assert name_of(first.id) == expected


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("top_k", [None, 25])
def test_random_scores_come_out_sorted_by_new_score(seed, top_k):
    """Randomized check against a plain sort, for several queue sizes."""
    rng = random.Random(seed)
    names_ = [f"obj{i}" for i in range(200)]
    old = {name: rng.random() for name in names_}
    new = {name: rng.random() for name in names_}

    if top_k is None:
        strategy = FullReexaminationStrategy(None)  # type: ignore[arg-type]
        reexamined = names_
    else:
        strategy = TopReexaminationStrategy(None, top_k)  # type: ignore[arg-type]
        reexamined = sorted(names_, key=lambda n: -old[n])[:top_k]

    new_queue, count = strategy.get_new_queues(FakeModel(new), make_queue(old))

    assert count == len(reexamined)
    assert names(drain(new_queue)) == sorted(reexamined, key=lambda n: -new[n])
