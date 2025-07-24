# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import queue
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import zmq
from logzero import logger

from ..classes import class_name_to_str
from ..detection import Detection
from ..hawkobject import HawkObject
from ..objectid import ObjectId
from ..ports import H2C_PORT, S2H_PORT
from ..proto import common_pb2
from ..proto.messages_pb2 import SendLabel, SendTile
from .label_utils import LabelSample
from .stats import (
    HAWK_LABELED_QUEUE_LENGTH,
    HAWK_UNLABELED_QUEUE_LENGTH,
    HAWK_UNLABELED_QUEUE_TIME,
    HAWK_UNLABELED_RECEIVED,
    HAWK_UNLABELED_RECEIVED_SCORE,
)

if TYPE_CHECKING:
    from pathlib import Path

    from prometheus_client import Gauge, Histogram, Summary


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

    @classmethod
    def from_msg(cls, msg: bytes, mission_dir: Path) -> UnlabeledResult:
        request = SendTile()
        request.ParseFromString(msg)

        objectId = ObjectId.from_protobuf(request.object_id)

        # the scout thinks it might have found something
        # filter out any negatives (and 0 confidence scores)
        detections = Detection.from_protobuf_list(request.inferenced)
        groundtruth = Detection.from_protobuf_list(request.groundtruth)

        # logger.debug(f"Received sample, inferenced scores {detections}")
        score = (
            max(detection.confidence for detection in detections) if detections else 0.0
        )

        oracle_media_types = [obj.media_type for obj in request.oracle_data]

        result = cls(
            objectId=objectId,
            scoutIndex=request.scoutIndex,
            model_version=request.version,
            score=score,
            oracle_items=oracle_media_types,
            detections=detections,
            groundtruth=groundtruth,
            novel_sample=request.novel_sample,
        )

        for index, _data in enumerate(request.oracle_data):
            data = HawkObject.from_protobuf(_data)

            image_dir = "images" if not request.novel_sample else "novel"
            image_path = result.content(mission_dir / image_dir, index=index)
            image_file = data.to_file(image_path, index=index, mkdirs=True)

            logger.info(f"SAVED {image_file} for {result.objectId}")

        if request.feature_vector:
            fv_path = result.content(mission_dir / "feature_vectors", suffix=".pt")
            feature_vector = HawkObject.from_protobuf(request.feature_vector)
            feature_vector.to_file(fv_path, mkdirs=True)

        return result


@dataclass
class HomeToScoutWorker:
    mission_id: str
    mission_dir: Path
    scout: str
    h2c_port: int
    zmq_context: zmq.Context = field(  # type: ignore[type-arg]
        default_factory=zmq.Context,
        repr=False,
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
        labels = [
            common_pb2.Detection(
                class_name=class_name_to_str(detection.class_name),
                confidence=1.0,
                coords=common_pb2.Region(
                    center_x=detection.x,
                    center_y=detection.y,
                    width=detection.w,
                    height=detection.h,
                ),
            )
            for detection in result.detections
        ]

        assert result.objectId is not None
        msg = SendLabel(
            object_id=result.objectId.to_protobuf(),
            scoutIndex=result.scoutIndex,
            labels=labels,
        ).SerializeToString()

        # update statistic first to avoid negative queue lengths when
        # the thread with queue.get and queue_length.dec gets to run first
        self.labeled_queue_length.inc()
        self.queue.put(msg)


@dataclass
class ScoutQueue:
    mission_id: str
    mission_dir: Path
    strategy: Strategy
    scouts: list[str]
    h2c_port: int = H2C_PORT
    coordinator: int | None = None

    zmq_context: zmq.Context = field(  # type: ignore[type-arg]
        default_factory=zmq.Context,
        repr=False,
    )
    label_queue: queue.PriorityQueue[tuple[float, UnlabeledResult]] = field(
        default_factory=queue.PriorityQueue,
    )
    to_scout: list[HomeToScoutWorker] = field(init=False)

    unlabeled_received: list[Summary] = field(init=False)
    unlabeled_queue_length: Gauge = field(init=False)
    unlabeled_queue_time: Histogram = field(init=False)

    def __post_init__(self) -> None:
        self.to_scout = [
            HomeToScoutWorker(
                mission_id=self.mission_id,
                mission_dir=self.mission_dir,
                scout=scout,
                h2c_port=self.h2c_port,
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
            mission=self.mission_id,
        )
        self.unlabeled_queue_time = HAWK_UNLABELED_QUEUE_TIME.labels(
            mission=self.mission_id,
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
            result = UnlabeledResult.from_msg(msg, self.mission_dir)

            self.unlabeled_received[result.scoutIndex].observe(len(msg))
            self.unlabeled_received_score[result.scoutIndex].observe(result.score)

            received += 1
            received_from_scout.update([result.scoutIndex])
            logger.debug(
                f"Received {result.scoutIndex} {result.objectId} {result.score}",
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
        label = "negative" if not result.detections else "positive"
        logger.info(f"Sending {label} to {result.scoutIndex}: {result.objectId}")
        self.to_scout[scout_index].put(result)
