# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import json
import queue
import threading
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

import zmq
from logzero import logger

from ..ports import S2H_PORT
from ..proto.messages_pb2 import SendTiles

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event, Semaphore
    from pathlib import Path


@dataclass
class Result:
    object_id: str
    scout_index: int
    score: float
    size: int
    data: bytes

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
        return cls(
            object_id=request.objectId,
            scout_index=request.scoutIndex,
            score=request.score,
            size=request.ByteSize(),
            data=request.attributes["thumbnail.jpeg"],
        )


class InboundProcess:
    def __init__(self, results_jsonl: Path, strategy: str, num_scouts: int) -> None:
        self.results_jsonl = results_jsonl
        self.strategy = strategy
        self.num_scouts = num_scouts

        # images are stored in a directory next to the results.jsonl file
        self.tile_dir = results_jsonl.parent / "images"
        self.tile_dir.mkdir()

    def scout_to_labeler(self, next_label: Semaphore | None, stop_event: Event) -> None:
        label_queue: queue.PriorityQueue[tuple[float, Result]] = queue.PriorityQueue()
        inq = threading.Thread(
            target=self.scout_to_home,
            kwargs={
                "label_queue": label_queue,
                "strategy": self.strategy,
                "num_scouts": self.num_scouts,
                "stop_event": stop_event,
            },
        )
        inq.start()

        self.home_to_labeler(label_queue, next_label, stop_event)

    @staticmethod
    def scout_to_home(
        label_queue: queue.PriorityQueue[tuple[float, Result]],
        strategy: str,
        num_scouts: int,
        stop_event: Event,
    ) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.bind(f"tcp://*:{S2H_PORT}")
        logger.info("Inbound Receive data started")

        receive_sequence = 0
        scout_received = [0] * num_scouts

        while not stop_event.is_set():
            msg = socket.recv()
            result = Result.from_msg(msg)

            receive_sequence += 1
            scout_received[result.scout_index] += 1
            logger.debug(
                f"Received {result.object_id} {result.scout_index} {result.score}"
            )

            if strategy in ["fifo-scout", "round-robin"]:
                priority: float = scout_received[result.scout_index]
            elif strategy in ["score", "top"]:
                priority = -result.score
            else:  # strategy in ["fifo-home", "fifo"]:
                priority = receive_sequence

            label_queue.put((priority, result))

    def home_to_labeler(
        self,
        label_queue: queue.PriorityQueue[tuple[float, Result]],
        next_label: Semaphore | None,
        stop_event: Event,
    ) -> None:
        count = 0

        with suppress(KeyboardInterrupt):
            while not stop_event.is_set():
                # block until the labeler is ready to accept more.
                # uses a timeout to periodically check if stop_event is set.
                if next_label is not None:
                    acquired = next_label.acquire(timeout=1)
                    if not acquired:
                        continue

                result = None
                while result is None and not stop_event.is_set():
                    with suppress(queue.Empty):
                        _, result = label_queue.get(timeout=1)
                if result is None:
                    break

                logger.info(
                    f"Labeling {result.object_id} {result.scout_index} {result.score}"
                )
                tile_jpeg = self.tile_dir.joinpath(f"{count:06}.jpeg")
                tile_jpeg.write_bytes(result.data)
                logger.info(f"SAVED TILE {tile_jpeg}")

                meta_json = result.to_json(index=count)
                with self.results_jsonl.open("a") as f:
                    f.write(f"{meta_json}\n")

                # logger.info(f"Meta: {count:06} {meta_json}")
                label_queue.task_done()

                count += 1
