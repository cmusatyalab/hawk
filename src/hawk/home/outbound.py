# SPDX-FileCopyrightText: 2022-2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Home Outbound process: sending labels from HOME to SCOUTS"""

from __future__ import annotations

import json
import queue
import threading
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

import zmq
from logzero import logger

from ..proto.messages_pb2 import LabelWrapper, SendLabels
from .utils import tailf

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event, Semaphore
    from pathlib import Path

    from .stats import LabelStats


@dataclass
class Label:
    object_id: str
    scout_index: int
    size: int
    image_label: str
    bounding_boxes: list[tuple[float, float, float, float]]

    def to_msg(self) -> bytes:
        bboxes = [f"{b[0]} {b[1]} {b[2]} {b[3]}" for b in self.bounding_boxes or []]
        label = LabelWrapper(
            objectId=self.object_id,
            scoutIndex=self.scout_index,
            imageLabel=str(self.image_label),
            boundingBoxes=bboxes,
        )
        return SendLabels(label=label).SerializeToString()

    @classmethod
    def from_json(cls, json_data: str) -> Label:
        data = json.loads(json_data)
        return cls(
            object_id=data["objectId"],
            scout_index=data["scoutIndex"],
            size=data["size"],
            image_label=data["imageLabel"],
            bounding_boxes=data["boundingBoxes"],
        )


class OutboundProcess:
    def __init__(
        self,
        scout_ips: list[str],
        h2c_port: int,
        labeled_jsonl: Path,
        coordinator: int | None = None,
    ) -> None:
        """Outbound messages from HOME to SCOUT

        API calls from home to scouts to to send tile labels from home to
        coordinator scout.
        Uses PUSH/PULL messaging protocol.
        The network is bandwidth constricted using FireQos.
        """
        self.scout_ips = scout_ips
        self.h2c_port = h2c_port
        self.labeled_jsonl = labeled_jsonl
        self.coordinator = coordinator

    def labeler_to_scout(
        self, next_label: Semaphore | None, stop_event: Event, labelstats: LabelStats
    ) -> None:
        """Worker process to collect and serialize labels from the labeler"""
        # start per-scout threads to send back labels
        with suppress(KeyboardInterrupt), zmq.Context() as zmq_context:
            scouts: list[queue.Queue[bytes]] = [queue.Queue() for _ in self.scout_ips]

            for index, scout in enumerate(self.scout_ips):
                # if there is a coordinator we don't need to run all worker threads...
                if self.coordinator is not None and index != self.coordinator:
                    continue

                scout_thread = threading.Thread(
                    target=self.home_to_scout,
                    kwargs={
                        "zmq_context": zmq_context,
                        "scout": scout,
                        "h2c_port": self.h2c_port,
                        "scout_queue": scouts[index],
                        "stop_event": stop_event,
                    },
                )
                scout_thread.start()

            self.next_label = next_label
            self.stop_event = stop_event
            self.labelstats = labelstats

            self.labeler_to_home(scouts)

    def labeler_to_home(self, scouts: list[queue.Queue[bytes]]) -> None:
        # make sure the file with labels exists
        self.labeled_jsonl.touch()

        negatives = positives = total_size = 0

        # read label results and forward to the scouts
        for label_json in tailf(self.labeled_jsonl, self.stop_event):
            if self.next_label is not None:
                self.next_label.release()

            label = Label.from_json(label_json)
            msg = label.to_msg()

            scout_index = self.coordinator or label.scout_index
            logger.info(
                f"Send labels {label.image_label}"
                f" {scout_index} {label.object_id} {len(msg)}"
            )
            scouts[scout_index].put(msg)

            # update stats
            if label.image_label in [None, "", "0"]:
                negatives += 1
            else:
                positives += 1
            total_size += label.size

            self.labelstats.update(positives, negatives, total_size)

    @staticmethod
    def home_to_scout(
        zmq_context: zmq.Context,  # type: ignore[type-arg]
        scout: str,
        h2c_port: int,
        scout_queue: queue.Queue[bytes],
        stop_event: Event,
    ) -> None:
        """Worker thread to send labels to individual scouts"""
        h2c_socket = zmq_context.socket(zmq.PUSH)
        h2c_socket.connect(f"tcp://{scout}:{h2c_port}")

        while not stop_event.is_set():
            # next label msg off queue, periodically check stop_event
            with suppress(queue.Empty):
                msg = scout_queue.get(timeout=1)
                h2c_socket.send(msg)
