# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Whether reexamined results count towards the top-k batch."""

from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from hawk.scout.selection.topk_selector import TopKSelector

pytestmark = pytest.mark.scout

REEXAMINED = 5


class FakeReexamination:
    """Puts REEXAMINED results back in the queue after every new model."""

    def get_new_queues(self, model, old_queue, start_time=0):
        new_queue: queue.PriorityQueue = queue.PriorityQueue()
        return new_queue, REEXAMINED


def batch_after_new_model(**kwargs) -> int:
    selector = TopKSelector(
        "test-mission",
        k=2,
        batch_size=10,
        countermeasure_threshold=0.5,
        total_countermeasures=0,
        reexamination_strategy=FakeReexamination(),  # type: ignore[arg-type]
        **kwargs,
    )
    selector._mission = SimpleNamespace(start_time=0.0)  # type: ignore[assignment]
    model = SimpleNamespace(version=1, is_oracle=lambda: False, train_examples={"1": 3})
    selector._new_model(model)  # type: ignore[arg-type]
    return selector._batch_added


def test_reexamined_count_by_default() -> None:
    assert batch_after_new_model() == REEXAMINED


def test_exclude_reexamined() -> None:
    assert batch_after_new_model(exclude_reexamined=True) == 0
