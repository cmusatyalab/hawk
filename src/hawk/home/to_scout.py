# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import io
import queue
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import zmq
from logzero import logger
from prometheus_client import Gauge, Histogram, Summary

from ..classes import class_name_to_str
from ..ports import H2C_PORT, S2H_PORT
from ..proto.messages_pb2 import BoundingBox, SendLabel, SendTile
from .label_utils import Detection, LabelSample, ObjectId
from .stats import (
    HAWK_LABELED_QUEUE_LENGTH,
    HAWK_UNLABELED_QUEUE_LENGTH,
    HAWK_UNLABELED_QUEUE_TIME,
    HAWK_UNLABELED_RECEIVED,
    HAWK_UNLABELED_RECEIVED_SCORE,
)

matplotlib.use("agg")


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

        # scout thinks it might have found some positives
        # filter out negatives (and 0 confidence scores)
        # cleanup and merge class scores for identical regions
        detections = list(
            Detection.merge_detections(
                Detection.from_boundingbox(
                    bbox.x, bbox.y, bbox.w, bbox.h, bbox.class_name, bbox.confidence
                )
                for bbox in request.boundingBoxes
                if bbox.confidence and bbox.class_name not in ["", "neg", "negative"]
            )
        )

        # logger.debug(f"Received sample, inferenced scores {detections}")
        score = max(detection.max_score for detection in detections)

        result = cls(
            objectId=ObjectId(request.objectId),
            scoutIndex=request.scoutIndex,
            model_version=request.version,
            score=score,
            detections=detections,
        )

        data = request.attributes["thumbnail.jpeg"]

        tile_jpeg = result.unique_name(mission_dir / "images", ".jpeg")
        tile_jpeg.parent.mkdir(exist_ok=True)

        if result.objectId.endswith(".npy"):  # for radar missions with .npy files
            result.gen_heatmap(tile_jpeg, data)
        else:
            tile_jpeg.write_bytes(data)
        logger.info(f"SAVED TILE {tile_jpeg} for {result.objectId}")

        if request.feature_vector:
            feature_vector = torch.load(io.BytesIO(request.feature_vector))

            fv_path = result.unique_name(mission_dir / "feature_vectors", ".pt")
            fv_path.parent.mkdir(exist_ok=True)
            torch.save(feature_vector, fv_path)

        return result

    @staticmethod
    def gen_heatmap(tile_path: Path, data_: bytes) -> None:
        with io.BytesIO(data_) as bytes_file:
            data = np.load(bytes_file, allow_pickle=True)
        plt.imshow(
            data.sum(axis=2).transpose(), cmap="viridis", interpolation="nearest"
        )
        plt.xticks([0, 16, 32, 48, 63], [-13, -6.5, 0, 6.5, 13], fontsize=8)
        plt.yticks([0, 64, 128, 192, 255], [50, 37.5, 25, 12.5, 0])
        plt.xlabel("velocity (m/s)")
        plt.ylabel("range (m)")
        # plt.title("RD Map")
        plt.savefig(tile_path, bbox_inches="tight")
        plt.close("all")


@dataclass
class HomeToScoutWorker:
    mission_id: str
    scout: str
    h2c_port: int
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
        bboxes = [
            BoundingBox(
                x=bbox.x,
                y=bbox.y,
                w=bbox.w,
                h=bbox.h,
                class_name=class_name_to_str(class_name),
                confidence=1.0,
            )
            for bbox in result.detections
            for class_name in bbox.scores
        ]
        msg = SendLabel(
            objectId=result.objectId,
            scoutIndex=result.scoutIndex,
            boundingBoxes=bboxes,
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
            result = UnlabeledResult.from_msg(msg, self.mission_dir)

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
        label = "negative" if not result.detections else "positive"
        logger.info(f"Sending {label} to {result.scoutIndex}: {result.objectId}")
        self.to_scout[scout_index].put(result)
