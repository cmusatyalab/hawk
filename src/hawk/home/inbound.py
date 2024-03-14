# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import json
import queue
import threading
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Tuple, cast

import zmq
from logzero import logger

from ..mission_config import MissionConfig
from ..ports import S2H_PORT
from ..proto.messages_pb2 import SendTiles
from .hawk_typing import MetaQueueType

if TYPE_CHECKING:
    InboundQueueType = queue.PriorityQueue[
        Tuple[float, Dict[str, Any], Dict[str, bytes]]
    ]


class InboundProcess:
    def __init__(
        self, tile_dir: Path, meta_dir: Path, configuration: MissionConfig
    ) -> None:
        self._tile_dir = tile_dir
        self._meta_dir = meta_dir
        self._token = False
        self._running = False
        self._save_attribute = "thumbnail.jpeg"
        self._count = 1

        # --- Extra token selector code to modify labeling process.
        self.configuration = configuration
        selector_field = self.configuration["selector"]
        if selector_field["type"] == "token":
            logger.info("In token")
            self._token = True
            init_samples = selector_field["token"]["initial_samples"]
            self._num_scouts = len(self.configuration.scouts)
            self.total_init_samples = int(init_samples) * int(self._num_scouts)
            self._label_time = int(selector_field["token"]["label_time"])
            self._sample_count = 0
            self._rotation_mode = selector_field["token"]["rotation"]
            self._global_priority_queue: InboundQueueType = queue.PriorityQueue()
            self._per_scout_priority_queues: list[InboundQueueType] = []
            for _i in range(self._num_scouts):
                self._per_scout_priority_queues.append(queue.PriorityQueue())
        # ---

    def receive_data(self, result_q: MetaQueueType, stop_event: Event) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.bind(f"tcp://*:{S2H_PORT}")
        logger.info("Inbound Receive data started")

        try:
            while not stop_event.is_set():
                msg = socket.recv()
                request = SendTiles()
                request.ParseFromString(msg)
                object_id = request.objectId
                parent_scout = request.scoutIndex
                score = request.score
                print(object_id, parent_scout)
                data_name = f"{self._count:06}"

                data = cast(Dict[str, bytes], request.attributes)
                byte_size = request.ByteSize()

                assert self._save_attribute in data

                temp_meta_data = {
                    "objectId": object_id,
                    "scoutIndex": parent_scout,
                    "score": score,
                    "size": byte_size,
                }

                if self._token:
                    logger.info(
                        f"\n\nReceived init sample from scout {parent_scout},"
                        f" total count is: {self._sample_count}"
                    )
                    if self._rotation_mode == "round-robin":
                        self._per_scout_priority_queues[parent_scout].put(
                            (-score, temp_meta_data, data)
                        )

                        # move this line to the toek thread that pops images
                        # from pri queues

                        # for num, i in enumerate(self._per_scout_priority_queues):
                        # logger.info(f"Length of Queue for Scout {num} is {i.qsize()}")
                    elif self._rotation_mode == "top":
                        self._global_priority_queue.put((-score, temp_meta_data, data))
                    # if aggregate priority queue length is less than total
                    # init samples, then continue
                    # otherwise, go ahead and get() and process with labeling process.
                    self._sample_count += 1
                    if self._sample_count == self.total_init_samples:
                        if self._rotation_mode == "round-robin":
                            pri_q = self._per_scout_priority_queues
                        else:
                            pri_q = [self._global_priority_queue]
                        token_th = threading.Thread(
                            target=self.token_thread,
                            args=(
                                pri_q,
                                result_q,
                                self._rotation_mode,
                                data_name,
                                stop_event,
                            ),
                        )
                        token_th.start()

                else:
                    self.write_push(result_q, temp_meta_data, data_name, data, 0)

        except (Exception, KeyboardInterrupt) as e:
            logger.error(e)

    def write_push(
        self,
        result_queue: MetaQueueType,
        temp_meta_data: dict[str, Any],
        data_name: str,
        data: dict[str, bytes],
        local_counter: int,
    ) -> None:
        if self._token:
            data_name = f"{local_counter:06}"
        meta_path = self._meta_dir / f"{data_name}.json"
        # logger.info("Meta path top of write push: {}".format(meta_path))

        # label = 1 if '/1/' in object_id else 0
        tile_jpeg = self._tile_dir.joinpath(data_name).with_suffix(".jpeg")
        with open(tile_jpeg, "wb") as f:
            img_array = data[self._save_attribute]
            f.write(bytearray(img_array))

        # need to modify these attributes to what is actually being pulled from
        # the queue.
        meta_data = {
            "objectId": temp_meta_data["objectId"],
            "scoutIndex": temp_meta_data["scoutIndex"],
            "score": temp_meta_data["score"],
            "size": temp_meta_data["size"],
        }
        # logger.info("Meta path:{} {}".format(meta_path, meta_data))
        with open(meta_path, "w") as f:
            json.dump(meta_data, f)

        logger.info(
            f"Received {temp_meta_data['objectId']}"
            f" {temp_meta_data['scoutIndex']}"
            f" {temp_meta_data['score']}"
        )
        logger.info(f"SAVING TILES {tile_jpeg}")
        result_queue.put(str(meta_path))
        self._count += 1

    def token_thread(
        self,
        pri_queue: list[InboundQueueType],
        result_queue: MetaQueueType,
        mode: str,
        data_name: str,
        stop_event: Event,
    ) -> None:
        local_counter = 0
        local_sample_counter = 0
        while not stop_event.is_set():
            if mode == "round-robin":
                home_scout_token = local_counter % self._num_scouts
                # logger.info(f"Home scout token is: {home_scout_token}")
                if pri_queue[home_scout_token].qsize() > 0:
                    total_sample = pri_queue[home_scout_token].get()
                    local_sample_counter += 1
                else:
                    continue
            elif mode == "top":
                total_sample = pri_queue[0].get()
            data = total_sample[2]
            meta_data = total_sample[1]
            # object_id = meta_data["objectId"]
            # parent_scout = meta_data["scoutIndex"]
            # score = meta_data["score"]
            # byte_size = meta_data["size"]
            local_counter += 1
            self.write_push(result_queue, meta_data, data_name, data, local_counter)
