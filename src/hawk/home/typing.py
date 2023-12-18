# SPDX-FileCopyrightText: 2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    MetaQueueType = Queue[str]
    LabelQueueType = Queue[str]
    StatsQueueType = Queue[Tuple[int, int, int]]
else:
    MetaQueueType = Queue
    LabelQueueType = Queue
    StatsQueueType = Queue


class Labeler:
    def start_labeling(
        self,
        input_q: MetaQueueType,
        result_q: LabelQueueType,
        stats_q: StatsQueueType,
        stop_event: Event,
    ) -> None:
        ...
