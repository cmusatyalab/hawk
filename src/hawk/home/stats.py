# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import Counter, Enum, Gauge, Histogram, Summary

if TYPE_CHECKING:
    from prometheus_client.samples import Sample

HAWK_MISSION_STATUS = Enum(
    "hawk_mission_status",
    "Current state of the mission",
    states=["starting", "configuring", "running", "stopped"],
    labelnames=["mission"],
)
HAWK_UNLABELED_RECEIVED = Summary(
    "hawk_unlabeled_received",
    "Size (_sum) and count (_count) of samples received from each scout",
    labelnames=["mission", "scout"],
)
HAWK_UNLABELED_QUEUE_LENGTH = Gauge(
    "hawk_unlabeled_queue_length",
    "Number of samples queued in priority queue before labeling",
    labelnames=["mission"],
)
HAWK_UNLABELED_QUEUE_TIME = Histogram(
    "hawk_unlabeled_queue_time",
    "How much time a sample spent in the priority queue before labeling",
    labelnames=["mission"],
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0),
)
HAWK_LABELER_QUEUED_LENGTH = Gauge(
    "hawk_labeler_queued_length",
    "Number of samples written to labeler waiting to be labeled",
    labelnames=["mission", "labeler"],
)
HAWK_LABELER_QUEUED_TIME = Histogram(
    "hawk_labeler_queued_time",
    "Time elapsed until a sample was labeled (seconds)",
    labelnames=["mission", "labeler"],
    buckets=(0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 25.0, 50.0, 75.0, 100.0),
)
HAWK_LABELED_OBJECTS = Counter(
    "hawk_labeled_objects",
    "Number of samples that were labeled",
    labelnames=["mission", "labeler", "label"],
)
HAWK_LABELED_QUEUE_LENGTH = Gauge(
    "hawk_labeled_queue_length",
    "Number of labels queued to be sent back to each scout",
    labelnames=["mission", "scout"],
)


def collect_metric_samples(stat: Counter | Gauge) -> list[Sample]:
    return [
        sample
        for metric in stat.collect()
        for sample in metric.samples
        if not sample.name.endswith("_created")
    ]


def collect_summary_total(stat: Histogram | Summary) -> int:
    return int(
        sum(
            sample.value
            for metric in stat.collect()
            for sample in metric.samples
            if sample.name.endswith("_sum")
        )
    )
