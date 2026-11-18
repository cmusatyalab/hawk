# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""The class-aware selector picks per class, with the same budget as topk."""

from __future__ import annotations

from collections import Counter
from typing import ClassVar, Iterable, Sequence

import pytest

from hawk import ObjectId
from hawk.classes import NEGATIVE_CLASS, ClassList, ClassName
from hawk.detection import Detection
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
from hawk.scout.selection.class_aware_selector import ClassAwareSelector
from hawk.scout.selection.topk_selector import TopKSelector

pytestmark = pytest.mark.scout

CLASSES = ["bent", "lying", "upright"]


def make_result(name: str, scores: dict[str, float]) -> ResultProvider:
    """A result scored like the dnn_classifier: one detection per class and a
    score that is the sum of the positive classes.
    """
    negative = 1.0 - sum(scores.values())
    bboxes = [Detection(class_name=NEGATIVE_CLASS, confidence=negative)]
    bboxes += [
        Detection(class_name=ClassName(c), confidence=scores.get(c, 0.0))
        for c in CLASSES
    ]
    object_id = ObjectId(f"/negative/collection/id/{name}")
    return ResultProvider(object_id, sum(scores.values()), bboxes, model_version=0)


def name_of(result: ResultProvider) -> str:
    return result.id.oid.rsplit("/", 1)[-1]


class FakeModel:
    train_examples: ClassVar[dict[str, int]] = {"0": 0, "1": 0}

    def __init__(self, scores: dict[str, dict[str, float]], version: int = 0) -> None:
        self.scores = scores
        self.version = version
        self.inferred: list[str] = []

    def is_oracle(self) -> bool:
        return False

    def infer(self, requests: Sequence[ObjectId]) -> Iterable[ResultProvider]:
        for object_id in requests:
            name = object_id.oid.rsplit("/", 1)[-1]
            self.inferred.append(name)
            yield make_result(name, self.scores[name])


class StubDataManager:
    def __init__(self) -> None:
        self.class_counts = type("Counts", (), {"counter": Counter()})()


class StubMission:
    start_time = 0.0

    def __init__(self) -> None:
        self.class_list = ClassList(CLASSES)
        self._data_manager = StubDataManager()
        self._time = 0.0

    def mission_time(self) -> float:
        self._time += 1.0
        return self._time

    def log(self, msg: str, end_t: float | None = None) -> None:
        pass


def make_selector(
    scheduler: str = "round_robin",
    k: int = 3,
    strategy=None,
    warm_start: int = 0,
) -> tuple[ClassAwareSelector, StubMission]:
    if strategy is None:
        strategy = NoReexaminationStrategy(None)  # type: ignore[arg-type]
    selector = ClassAwareSelector(
        "test-mission", k, 1000, scheduler, 0.5, 50, strategy, warm_start=warm_start
    )
    mission = StubMission()
    selector.add_context(mission)  # type: ignore[arg-type]
    selector.new_model(FakeModel({}))  # type: ignore[arg-type]
    return selector, mission


def sent(selector: TopKSelector) -> list[str]:
    names = []
    while not selector.result_queue.empty():
        result = selector.result_queue.get_nowait()
        assert result is not None
        names.append(name_of(result))
    return names


# three upright-looking results, one of them a bit bent, and one result that
# only the lying score picks out
SCORES = {
    "upright1": {"upright": 0.9},
    "upright2": {"upright": 0.7, "bent": 0.1},
    "upright3": {"upright": 0.6},
    "maybe_lying": {"lying": 0.3},
}


def add_all(selector: TopKSelector, scores: dict[str, dict[str, float]]) -> None:
    for name, class_scores in scores.items():
        selector.add_result(make_result(name, class_scores))


def test_topk_misses_the_rare_class():
    selector = TopKSelector(
        "test-mission",
        3,
        1000,
        0.5,
        50,
        NoReexaminationStrategy(None),  # type: ignore[arg-type]
    )
    selector.add_context(StubMission())  # type: ignore[arg-type]
    selector.new_model(FakeModel({}))  # type: ignore[arg-type]
    add_all(selector, SCORES)

    selector.select_tiles(3)
    assert sent(selector) == ["upright1", "upright2", "upright3"]


def test_round_robin_takes_one_pick_per_class():
    selector, _ = make_selector("round_robin")
    add_all(selector, SCORES)

    selector.select_tiles(3)
    # bent queue: upright2 has the only bent score, lying queue: maybe_lying,
    # upright queue: upright1
    assert sent(selector) == ["upright2", "maybe_lying", "upright1"]


def test_round_robin_continues_where_it_left_off():
    selector, _ = make_selector("round_robin")
    add_all(selector, SCORES)

    selector.select_tiles(1)
    selector.select_tiles(1)
    assert sent(selector) == ["upright2", "maybe_lying"]


def test_a_result_is_never_sent_twice():
    selector, _ = make_selector("round_robin")
    add_all(selector, SCORES)

    selector.select_tiles(4)
    selector.select_tiles(4)
    names = sent(selector)
    assert sorted(names) == sorted(SCORES)


def test_deficit_picks_the_class_with_fewest_labels():
    selector, mission = make_selector("deficit", k=1)
    mission._data_manager.class_counts.counter.update(
        {"bent": 5, "lying": 1, "upright": 9},
    )
    add_all(selector, SCORES)

    selector.select_tiles(1)
    assert sent(selector) == ["maybe_lying"]


def test_deficit_spreads_the_picks_of_one_batch():
    selector, mission = make_selector("deficit")
    mission._data_manager.class_counts.counter.update(
        {"bent": 2, "lying": 1, "upright": 2},
    )
    add_all(selector, SCORES)

    # lying has the fewest labels, after that ties go in class list order
    selector.select_tiles(3)
    assert sent(selector) == ["maybe_lying", "upright2", "upright1"]


def test_unknown_scheduler():
    with pytest.raises(ValueError, match="scheduler"):
        make_selector("most_confident")


@pytest.mark.parametrize(
    "strategy",
    [FullReexaminationStrategy(None), TopReexaminationStrategy(None, 2)],  # type: ignore[arg-type]
    ids=["full", "top"],
)
def test_reexamination_ranks_each_class_by_the_new_model(strategy):
    selector, _ = make_selector("round_robin", strategy=strategy)
    # a little lying score puts upright3 in the top 2 of the lying queue
    old_scores = dict(SCORES, upright3={"upright": 0.6, "lying": 0.05})
    add_all(selector, old_scores)

    # the new model thinks upright3 is the lying one
    new_scores = dict(old_scores)
    new_scores["upright3"] = {"lying": 0.6, "upright": 0.3}
    new_scores["maybe_lying"] = {"upright": 0.95}
    model = FakeModel(new_scores, version=1)
    selector.new_model(model)  # type: ignore[arg-type]

    # every object is inferred once, however many class queues it is in
    assert sorted(model.inferred) == sorted(set(model.inferred))

    selector._next_class = 1  # lying
    selector.select_tiles(1)
    result = selector.result_queue.get_nowait()
    assert result is not None
    assert name_of(result) == "upright3"
    # what gets sent is the full result, scored by the sum of the classes
    assert result.score == pytest.approx(0.9)


def test_reexamination_counts_towards_the_batch_like_topk():
    strategy = TopReexaminationStrategy(None, 2)  # type: ignore[arg-type]
    selector, _ = make_selector("round_robin", strategy=strategy)
    add_all(selector, SCORES)
    before = selector._batch_added

    selector.new_model(FakeModel(SCORES, version=1))  # type: ignore[arg-type]

    # top 2 of each of the 3 class queues are rescored, but the batch only
    # counts the 2 a single topk queue would have rescored
    assert selector._batch_added - before == 2


def test_reexamination_drops_sent_results():
    strategy = FullReexaminationStrategy(None)  # type: ignore[arg-type]
    selector, _ = make_selector("round_robin", strategy=strategy)
    add_all(selector, SCORES)
    selector.select_tiles(2)
    first = sent(selector)

    selector.new_model(FakeModel(SCORES, version=1))  # type: ignore[arg-type]
    selector.select_tiles(4)
    rest = sent(selector)

    assert not set(first) & set(rest)
    assert sorted(first + rest) == sorted(SCORES)


def test_easy_negatives_see_each_queued_result_once():
    selector, _ = make_selector("round_robin")
    add_all(selector, SCORES)
    selector.select_tiles(1)

    queued = [name_of(result) for result in selector._queued_results()]
    assert sorted(queued) == sorted(set(SCORES) - {"upright2"})


def test_warm_start_picks_like_topk():
    selector, _ = make_selector("round_robin", warm_start=1)
    add_all(selector, SCORES)

    selector.select_tiles(3)
    assert sent(selector) == ["upright1", "upright2", "upright3"]


def test_warm_start_ends_at_the_model_version():
    strategy = FullReexaminationStrategy(None)  # type: ignore[arg-type]
    selector, _ = make_selector("round_robin", strategy=strategy, warm_start=1)
    add_all(selector, SCORES)
    selector.select_tiles(1)
    first = sent(selector)

    selector.new_model(FakeModel(SCORES, version=1))  # type: ignore[arg-type]
    selector.select_tiles(3)
    rest = sent(selector)

    # the class queues take over, upright1 is not sent again
    assert first == ["upright1"]
    assert rest == ["upright2", "maybe_lying", "upright3"]


def test_warm_start_ranks_by_the_new_model():
    strategy = FullReexaminationStrategy(None)  # type: ignore[arg-type]
    selector, _ = make_selector("round_robin", strategy=strategy, warm_start=2)
    add_all(selector, SCORES)

    new_scores = dict(SCORES, maybe_lying={"lying": 0.95})
    selector.new_model(FakeModel(new_scores, version=1))  # type: ignore[arg-type]
    selector.select_tiles(2)
    assert sent(selector) == ["maybe_lying", "upright1"]
