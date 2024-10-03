# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import json
import queue
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

import zmq
from logzero import logger
from prometheus_client import Gauge, Histogram, Summary

from ..ports import H2C_PORT, S2H_PORT
from ..proto.messages_pb2 import LabelWrapper, SendLabels, SendTiles
from .label_utils import BoundingBox, ClassMap, LabelSample
from .stats import (
    HAWK_LABELED_QUEUE_LENGTH,
    HAWK_UNLABELED_QUEUE_LENGTH,
    HAWK_UNLABELED_QUEUE_TIME,
    HAWK_UNLABELED_RECEIVED,
    HAWK_UNLABELED_RECEIVED_SCORE,
)


class Strategy(Enum):
    FIFO_HOME = "fifo-home"
    FIFO = "fifo"
    FIFO_SCOUT = "fifo-scout"
    ROUND_ROBIN = "round-robin"
    SCORE = "score"
    TOP = "top"

    def __str__(self) -> str:
        return self.value


@dataclass
class UnlabeledResult(LabelSample):
    score: float = 1.0
    data: bytes = b""

    @classmethod
    def from_msg(cls, msg: bytes, class_map: ClassMap) -> UnlabeledResult:
        request = SendTiles()
        request.ParseFromString(msg)

        if "boxes" in request.attributes.keys():
            bb_string = request.attributes["boxes"].decode()
            bounding_boxes = [
                BoundingBox.from_yolo(line, class_map) for line in json.loads(bb_string)
            ]
        else:
            assert "scores" in request.attributes
            # classification case, scout thinks this could be a positive
            scores = json.loads(request.attributes["scores"])
            logger.debug(f"Received sample, inferenced scores {scores}")
            bounding_boxes = [
                BoundingBox(label=label, score=score)
                for label, score in scores.items()
                if score and label != "negative"
            ]

        return cls(
            objectId=request.objectId,
            scoutIndex=request.scoutIndex,
            score=request.score,
            labels=bounding_boxes,
            data=request.attributes["thumbnail.jpeg"],
        )


@dataclass
class HomeToScoutWorker:
    mission_id: str
    scout: str
    h2c_port: int
    class_map: ClassMap
    zmq_context: zmq.Context = field(  # type: ignore[type-arg]
        default_factory=zmq.Context, repr=False
    )
    queue: queue.SimpleQueue[bytes] = field(default_factory=queue.SimpleQueue)

    labeled_queue_length: Gauge = field(init=False)

    def __post_init__(self) -> None:
        self.labeled_queue_length = HAWK_LABELED_QUEUE_LENGTH.labels(
            mission=self.mission_id,
            scout=self.scout,
        )
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        """Worker thread to send label results back to the scout."""
        h2c_socket = self.zmq_context.socket(zmq.PUSH)
        h2c_socket.connect(f"tcp://{self.scout}:{self.h2c_port}")
        logger.info(f"Outbound send to {self.scout} started")

        while True:
            msg = self.queue.get()
            self.labeled_queue_length.dec()
            h2c_socket.send(msg)

    def put(self, result: LabelSample) -> None:
        """Queue a label from any thread which will be sent to the scout."""
        imageLabel = "0" if not result.labels else "1"
        bboxes = [bbox.to_yolo(self.class_map) for bbox in result.labels]
        label = LabelWrapper(
            objectId=result.objectId,
            scoutIndex=result.scoutIndex,
            imageLabel=imageLabel,
            boundingBoxes=bboxes,
        )
        msg = SendLabels(label=label).SerializeToString()

        # update statistic first to avoid negative queue lengths when
        # the thread with queue.get and queue_length.dec gets to run first
        self.labeled_queue_length.inc()
        self.queue.put(msg)


@dataclass
class ScoutQueue:
    mission_id: str
    strategy: Strategy
    scouts: list[str]
    class_map: ClassMap
    h2c_port: int = H2C_PORT
    coordinator: int | None = None

    zmq_context: zmq.Context = field(  # type: ignore[type-arg]
        default_factory=zmq.Context, repr=False
    )
    label_queue: queue.PriorityQueue[tuple[float, UnlabeledResult]] = field(
        default_factory=queue.PriorityQueue
    )
    to_scout: list[HomeToScoutWorker] = field(init=False)

    unlabeled_received: list[Summary] = field(init=False)
    unlabeled_queue_length: Gauge = field(init=False)
    unlabeled_queue_time: Histogram = field(init=False)

    def __post_init__(self) -> None:
        self.to_scout = [
            HomeToScoutWorker(
                mission_id=self.mission_id,
                scout=scout,
                h2c_port=self.h2c_port,
                class_map=self.class_map,
                zmq_context=self.zmq_context,
            )
            for scout in self.scouts
        ]
        self.unlabeled_received = [
            HAWK_UNLABELED_RECEIVED.labels(mission=self.mission_id, scout=scout)
            for scout in self.scouts
        ]
        self.unlabeled_received_score = [
            HAWK_UNLABELED_RECEIVED_SCORE.labels(mission=self.mission_id, scout=scout)
            for scout in self.scouts
        ]
        self.unlabeled_queue_length = HAWK_UNLABELED_QUEUE_LENGTH.labels(
            mission=self.mission_id
        )
        self.unlabeled_queue_time = HAWK_UNLABELED_QUEUE_TIME.labels(
            mission=self.mission_id
        )

    def start(self) -> ScoutQueue:
        threading.Thread(target=self.scout_to_home, daemon=True).start()
        return self

    def run(self) -> None:
        self.scout_to_home()

    def scout_to_home(self) -> None:
        socket = self.zmq_context.socket(zmq.PULL)
        socket.bind(f"tcp://*:{S2H_PORT}")
        logger.info("Inbound receive from scouts started")

        received = 0
        received_from_scout: Counter[int] = Counter()

        while True:
            msg = socket.recv()
            result = UnlabeledResult.from_msg(msg, self.class_map)

            self.unlabeled_received[result.scoutIndex].observe(len(msg))
            self.unlabeled_received_score[result.scoutIndex].observe(result.score)

            received += 1
            received_from_scout.update([result.scoutIndex])
            logger.debug(
                f"Received {result.scoutIndex} {result.objectId} {result.score}"
            )

            if self.strategy in [Strategy.SCORE, Strategy.TOP]:
                priority = -result.score
            elif self.strategy in [Strategy.FIFO_SCOUT, Strategy.ROUND_ROBIN]:
                priority = received_from_scout[result.scoutIndex]
            else:  # self.strategy in [Strategy.FIFO_HOME, Strategy.FIFO]:
                priority = received

            self.unlabeled_queue_length.inc()
            self.label_queue.put((priority, result))

    def get(self) -> UnlabeledResult:
        _, result = self.label_queue.get()
        self.unlabeled_queue_length.dec()
        self.unlabeled_queue_time.observe(time.time() - result.queued)
        return result

    def task_done(self) -> None:
        self.label_queue.task_done()

    def put(self, result: LabelSample) -> None:
        """Queue a label to be sent back to the scout.

        Will be sent to the scout that sent the original result or the coordinator.
        """
        scout_index = self.coordinator if self.coordinator else result.scoutIndex
        label = "negative" if not result.labels else "positive"
        logger.info(f"Sending {label} to {result.scoutIndex}: {result.objectId}")
        self.to_scout[scout_index].put(result)
