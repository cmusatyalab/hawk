# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import json
import queue
import threading
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

import zmq
from logzero import logger

from ..ports import H2C_PORT, S2H_PORT
from ..proto.messages_pb2 import LabelWrapper, SendLabels, SendTiles


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
class Result:
    object_id: str
    scout_index: int
    score: float
    size: int
    data: bytes
    inferred_boxes: list[tuple[float, float, float, float]]

    def to_json(self, **kwargs: int | float | str) -> str:
        return json.dumps(
            dict(
                objectId=self.object_id,
                scoutIndex=self.scout_index,
                score=self.score,
                size=self.size,
                **kwargs,
            ),
            sort_keys=True,
        )

    @classmethod
    def from_msg(cls, msg: bytes) -> Result:
        request = SendTiles()
        request.ParseFromString(msg)

        if 'boxes' in request.attributes.keys():
            bb_string = request.attributes['boxes'].decode()
            bounding_boxes = json.loads(bb_string)
        else:
            bounding_boxes = []

        return cls(
            object_id=request.objectId,
            scout_index=request.scoutIndex,
            score=request.score,
            size=request.ByteSize(),
            data=request.attributes["thumbnail.jpeg"],
            inferred_boxes=bounding_boxes,
        )


@dataclass
class Label:
    object_id: str
    scout_index: int
    size: int
    image_label: str
    bounding_boxes: list[tuple[float, float, float, float, float]]
    queued_time: float | None = None

    def to_msg(self) -> bytes:
        bboxes = [f"{int(b[0])} {b[1]} {b[2]} {b[3]} {b[4]}" for b in self.bounding_boxes or []]
        label = LabelWrapper(
            objectId=self.object_id,
            scoutIndex=self.scout_index,
            imageLabel=str(self.image_label),
            boundingBoxes=bboxes,
        )
        return SendLabels(label=label).SerializeToString()

    def to_json(self, **kwargs: int | float | str) -> str:
        return json.dumps(
            dict(
                objectId=self.object_id,
                scoutIndex=self.scout_index,
                size=self.size,
                imageLabel=self.image_label,
                boundingBoxes=self.bounding_boxes,
                **kwargs,
            ),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, json_data: str) -> Label:
        data = json.loads(json_data)
        return cls(
            object_id=data["objectId"],
            scout_index=data["scoutIndex"],
            size=data["size"],
            image_label=data["imageLabel"],
            bounding_boxes=data["boundingBoxes"],
            queued_time=data.get("queued_time"),
        )


@dataclass
class HomeToScoutWorker:
    scout: str
    h2c_port: int
    zmq_context: zmq.Context = field(  # type: ignore[type-arg]
        default_factory=zmq.Context, repr=False
    )
    queue: queue.SimpleQueue[bytes] = field(default_factory=queue.SimpleQueue)

    def __post_init__(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        """Worker thread to send label results back to the scout."""
        h2c_socket = self.zmq_context.socket(zmq.PUSH)
        h2c_socket.connect(f"tcp://{self.scout}:{self.h2c_port}")
        logger.info(f"Outbound send to {self.scout} started")

        while True:
            msg = self.queue.get()
            h2c_socket.send(msg)

    def put(self, label: Label) -> None:
        """Queue a label from any thread which will be sent to the scout."""
        msg = label.to_msg()
        self.queue.put(msg)


@dataclass
class ScoutQueue:
    strategy: Strategy
    scouts: list[str]
    h2c_port: int = H2C_PORT
    coordinator: int | None = None

    zmq_context: zmq.Context = field(  # type: ignore[type-arg]
        default_factory=zmq.Context, repr=False
    )
    label_queue: queue.PriorityQueue[tuple[float, Result]] = field(
        default_factory=queue.PriorityQueue
    )
    to_scout: list[HomeToScoutWorker] = field(init=False)

    def __post_init__(self) -> None:
        self.to_scout = [
            HomeToScoutWorker(
                scout=scout, h2c_port=self.h2c_port, zmq_context=self.zmq_context
            )
            for scout in self.scouts
        ]

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
            result = Result.from_msg(msg)

            received += 1
            received_from_scout.update([result.scout_index])
            logger.debug(
                f"Received {result.scout_index} {result.object_id} {result.score}"
            )

            if self.strategy in [Strategy.SCORE, Strategy.TOP]:
                priority = -result.score
            elif self.strategy in [Strategy.FIFO_SCOUT, Strategy.ROUND_ROBIN]:
                priority = received_from_scout[result.scout_index]
            else:  # self.strategy in [Strategy.FIFO_HOME, Strategy.FIFO]:
                priority = received

            self.label_queue.put((priority, result))

    def get(self) -> Result:
        _, result = self.label_queue.get()
        return result

    def task_done(self) -> None:
        self.label_queue.task_done()

    def put(self, label: Label) -> None:
        """Queue a label to be sent back to the scout.

        Will be sent to the scout that sent the original result or the coordinator.
        """
        scout_index = (
            label.scout_index if self.coordinator is None else self.coordinator
        )
        logger.info(
            f"Sending label {label.image_label} {scout_index} {label.object_id}"
        )
        self.to_scout[scout_index].put(label)
