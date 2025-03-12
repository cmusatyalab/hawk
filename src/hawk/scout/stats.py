# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HAWK_RETRIEVER_TOTAL_IMAGES = Gauge(
    "hawk_retriever_total_images",
    "Total number of images in mission",
    labelnames=["mission"],
)
HAWK_RETRIEVER_TOTAL_OBJECTS = Gauge(
    "hawk_retriever_total_objects",
    "Total number of samples in mission",
    labelnames=["mission"],
)
HAWK_RETRIEVER_RETRIEVED_IMAGES = Counter(
    "hawk_retriever_retrieved_images",
    "Number of images retrieved",
    labelnames=["mission"],
)
HAWK_RETRIEVER_RETRIEVED_OBJECTS = Counter(
    "hawk_retriever_retrieved_objects",
    "Number of samples retrieved",
    labelnames=["mission"],
)
HAWK_RETRIEVER_FAILED_OBJECTS = Counter(
    "hawk_retriever_failed_objects",
    "Number of samples we failed to read",
    labelnames=["mission"],
)
HAWK_RETRIEVER_DROPPED_OBJECTS = Counter(
    "hawk_retriever_dropped_objects",
    "Number of samples dropped (retriever queue full)",
    labelnames=["mission"],
)
HAWK_RETRIEVER_QUEUE_LENGTH = Gauge(
    "hawk_retriever_queue_length",
    "Number of tiles queued for inferencing",
    labelnames=["mission"],
)

HAWK_MODEL_VERSION = Gauge(
    "hawk_model_version",
    "Gauge to track the trained model iteration used for inference",
    labelnames=["mission"],
)

HAWK_INFERENCED_OBJECTS = Histogram(
    "hawk_inferenced_objects",
    "Histogram to track confidence scores of inferenced objects (count/sum/buckets)",
    labelnames=["mission", "gt", "model_version"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

HAWK_SELECTOR_SKIPPED_OBJECTS = Counter(
    "hawk_selector_skipped_objects",
    "Number of samples that skipped the selector because there was no model",
    labelnames=["mission"],
)
HAWK_SELECTOR_DISCARD_QUEUE_LENGTH = Gauge(
    "hawk_threshold_selector_discard_queue_length",
    "Number of samples queued for reexamination by the threshold selector",
    labelnames=["mission"],
)
HAWK_SELECTOR_DROPPED_OBJECTS = Counter(
    "hawk_threshold_selector_dropped_objects",
    "Number of samples dropped by the threshold selector",
    labelnames=["mission"],
)
HAWK_SELECTOR_FALSE_NEGATIVES = Counter(
    "hawk_threshold_selector_false_negatives",
    "Number of false negatives dropped by the threshold selector",
    labelnames=["mission"],
)
HAWK_SELECTOR_PRIORITY_QUEUE_LENGTH = Gauge(
    "hawk_selector_priority_queue_length",
    "Number of samples queued in the selector priority queue(s)",
    labelnames=["mission"],
)
HAWK_SELECTOR_REVISITED_OBJECTS = Counter(
    "hawk_selector_revisited_objects",
    "Number of reexamined samples",
    labelnames=["mission"],
)
HAWK_SELECTOR_DEQUEUED_OBJECTS = Histogram(
    "hawk_selector_dequeued_objects",
    "Histogram to track confidence scores of objects sent to home (count/sum/buckets)",
    labelnames=["mission", "gt", "model_version"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
HAWK_SELECTOR_RESULT_QUEUE_LENGTH = Gauge(
    "hawk_selector_result_queue_length",
    "Number of samples queued for sending to hawk_home",
    labelnames=["mission"],
)

HAWK_SURVIVABILITY_TRUE_POSITIVES = Counter(
    "hawk_survivability_true_positive",
    "True positive based on the survivability countermeasure threshold",
    labelnames=["mission"],
)
HAWK_SURVIVABILITY_FALSE_POSITIVES = Counter(
    "hawk_survivability_false_positive",
    "False positive based on the survivability countermeasure threshold",
    labelnames=["mission"],
)
HAWK_SURVIVABILITY_FALSE_NEGATIVES = Counter(
    "hawk_survivability_false_negative",
    "False negative based on the survivability countermeasure threshold",
    labelnames=["mission"],
)
HAWK_SURVIVABILITY_TRUE_NEGATIVES = Counter(
    "hawk_survivability_true_negative",
    "True negative based on the survivability countermeasure threshold",
    labelnames=["mission"],
)
HAWK_SURVIVABILITY_THREATS_NOT_COUNTERED = Counter(
    "hawk_survivability_threats_not_countered",
    "Survivability threats not countered (FN + TP after depleting CMs)",
    labelnames=["mission"],
)


# duplicate with hawk.home.stats maybe move to hawk.stats or hawk.utils?
def collect_metrics_total(metric: Counter | Gauge) -> int:
    return int(
        sum(
            sample.value
            for instance in metric.collect()
            for sample in instance.samples
            if not sample.name.endswith("_created")
        )
    )
