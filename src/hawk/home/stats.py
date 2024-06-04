# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Summary

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
HAWK_LABELED_POSITIVE = Counter(
    "hawk_labeled_positive",
    "Number of samples that were labeled as True Positive",
    labelnames=["mission", "labeler"],
)
HAWK_LABELED_NEGATIVE = Counter(
    "hawk_labeled_negative",
    "Number of samples that were labeled as False Positive",
    labelnames=["mission", "labeler"],
)
HAWK_LABELED_QUEUE_LENGTH = Gauge(
    "hawk_labeled_queue_length",
    "Number of labels queued to be sent back to each scout",
    labelnames=["mission", "scout"],
)


def collect_counter_total(counter: Counter) -> float:
    return sum(
        sample.value
        for metric in counter.collect()
        for sample in metric.samples
        if not sample.name.endswith("_created")
    )


def collect_summary_total(summary: Summary) -> float:
    return sum(
        sample.value
        for metric in summary.collect()
        for sample in metric.samples
        if sample.name.endswith("_sum")
    )
