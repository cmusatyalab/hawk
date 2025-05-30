# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import io
import math
import queue
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import zmq
from logzero import logger
from PIL import Image

from ...classes import NEGATIVE_CLASS, ClassLabel, ClassName
from ...objectid import ObjectId
from ...proto.messages_pb2 import NetworkDataset
from ..core.object_provider import ObjectProvider
from ..stats import collect_metrics_total
from .retriever import Retriever


class NetworkRetriever(Retriever):
    def __init__(self, mission_id: str, dataset: NetworkDataset, host_name: str):
        globally_constant_rate = dataset.dataBalanceMode == "globally_constant"
        super().__init__(
            mission_id,
            tiles_per_interval=dataset.numTiles,
            globally_constant_rate=globally_constant_rate,
        )
        self.this_host_name = host_name.split(":")[0]
        self._dataset = dataset
        self._network_server_host = dataset.dataServerAddr  ## pulled from home config
        self._network_server_port = dataset.dataServerPort  ## pulled from home config
        # True unless local scout deploy conditions prevent
        self.scml_active_condition = True
        self.data_rate_balance = dataset.dataBalanceMode

        self._timeout = dataset.timeout  ## from home config default 20
        self._resize = dataset.resizeTile
        logger.info("In NETWORK RETRIEVER INIT...")

        index_file = Path(self._dataset.dataPath)
        self._data_root = index_file.parent.parent
        self.contents = index_file.read_text().splitlines()
        self.total_tiles = len(self.contents)
        ## for network retriever, will update total_tiles to equal tiles
        ## processes as clients unaware of total tiles per scout a priori

        key_len = math.ceil(self.total_tiles / self.tiles_per_interval)

        keys = np.arange(key_len)
        per_frame = np.array_split(self.contents, key_len)

        self.img_tile_map = defaultdict(list)
        for i, tiles_per_frame in enumerate(per_frame):
            k = keys[i]
            for content in tiles_per_frame:
                self.img_tile_map[k].append(content)

        # random.shuffle(keys)
        self.images = keys

        self.total_images.set(len(self.images))
        if self._network_server_host == self.this_host_name:
            self.total_objects.set(self.total_tiles)
        else:
            self.total_objects.set(0)
        ## allow only serve to report how many samples have been processed.

        self.request_counter_by_scout: dict[int, int] = {}
        self.sample_count = 0

    def _run_threads(self) -> None:
        if self._network_server_host == self.this_host_name:
            logger.info("Starting server thread in retriever...")
            threading.Thread(target=self.server, name="network_server").start()
        else:
            super()._run_threads()

    def stream_objects(self) -> None:
        super().stream_objects()
        assert self._context is not None

        # network retriever remains idle until we get the network connection
        # up and the scml conditions are satisfied.
        self.current_deployment_mode = "Idle"

        context = zmq.Context()
        client_socket = context.socket(zmq.REQ)
        client_socket.connect(
            f"tcp://{self._network_server_host}:{self._network_server_port}"
        )  ## setup socket

        ## next will need to manage some flags for when network sample
        ## retrieval stop and starts, otherwise the net retriever should
        ## continuously send requests to the server.
        scout_index = str(self._context.scout_index).encode()
        time_start = time.time()
        mission_time_start = time.time()
        logger.info(f"SCML options from mission: {self._context.scml_deploy_options}")
        while True:
            ## if sent deployment order from home or preset conditions are reached.
            if self.scml_active_mode is not None:
                active = self.scml_active_mode
            else:
                mission_time = time.time() - mission_time_start
                model_version = self._context.model_version
                scml_deploy_options = self._context.scml_deploy_options

                # when start conditions are specified we should only switch to
                # active state when any of the start conditions are triggered.
                # Otherwise we assume we begin from an active state that may
                # get disabled by any not_before_ or end_ conditions.
                start_conditions = {"start_time", "start_on_model"}
                active = not bool(start_conditions.intersection(scml_deploy_options))

                if "start_time" in scml_deploy_options:
                    active |= mission_time >= scml_deploy_options["start_time"]

                if "start_on_model" in scml_deploy_options:
                    active |= model_version >= scml_deploy_options["start_on_model"]

                if "not_before_time" in scml_deploy_options:
                    active &= mission_time >= scml_deploy_options["not_before_time"]
                if "not_before_model" in scml_deploy_options:
                    active &= model_version >= scml_deploy_options["not_before_model"]

                if "end_time" in scml_deploy_options:
                    active &= mission_time < scml_deploy_options["end_time"]
                if "end_on_model" in scml_deploy_options:
                    active &= model_version < scml_deploy_options["end_on_model"]

            self.current_deployment_mode = "Active" if active else "Idle"

            ## i.e. if it's Dead or Idle, 'Dead' not fully implemented yet
            if not active:
                logger.info(
                    "Too early to start retrieving or in idle mode... "
                    f"{time.time() - mission_time_start}"
                )
                time.sleep(5)
                continue
            ## need additional conditions here to break out of this loop, or to
            ## block at some line before retrieving the next sample.
            ## prepare and send request
            self.sample_count += 1
            self.total_tiles = self.sample_count
            ## next step will need the server to send the number of remaining
            ## samples on server, which should decrease over time.
            msg = [scout_index, str(self.sample_count).encode()]
            client_socket.send_multipart(msg)

            ### receive and process
            content, label_, path_ = client_socket.recv_multipart()
            label, path = label_.decode(), path_.decode()

            # normalize labels from int/str to ClassName
            try:
                class_label = ClassLabel(int(label))
                class_name = self._class_id_to_name(class_label)
            except ValueError:
                class_name = ClassName(sys.intern(label))
            except IndexError:
                class_name = NEGATIVE_CLASS

            object_id = ObjectId(f"/{class_name}/collection/id/{path}")

            self.put_objects(
                ObjectProvider(
                    object_id,
                    content,
                    class_name,
                )
            )

            ## XXX The following logic/sleep loop should probably move into
            ## the Retriever base class to avoid unnecessary code duplication.
            if not self.globally_constant_rate:
                num_tiles = self.tiles_per_interval
            elif self.active_scout_ratio == 0.0:
                num_tiles = self.sample_count  # avoid divide by 0, just sleep
            else:  # adjust local retrieval rate to compensate for lost scouts
                num_tiles = int(self.tiles_per_interval / self.active_scout_ratio)

            if self.sample_count % num_tiles == 0:
                retrieved_tiles = collect_metrics_total(self.retrieved_objects)
                logger.info(f"{retrieved_tiles} / {self.total_tiles} RETRIEVED")
                logger.info(
                    f"Num tiles, {num_tiles}, "
                    f"tiles per interval: {self.tiles_per_interval}, "
                    f"active scout ratio: {self.active_scout_ratio}"
                )
                time_passed = time.time() - time_start
                if time_passed < self._timeout:
                    logger.info(f"About to sleep at: {time.time()}")
                    time.sleep(self._timeout - time_passed)
                time_start = time.time()
            ## may need to adjust this timeout of 20 to accomodate higher
            ## retrieval rates.

    def server(self) -> None:
        self.current_deployment_mode = "Server"
        served_samples = 0
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://0.0.0.0:{self._network_server_port}")  ## setup socket
        logger.info("Server socket ready...")
        sample_queue: queue.Queue[tuple[Path, str]] = queue.Queue()
        for line in self.contents:
            path, label = line.split()
            sample_queue.put((Path(path), label))
        logger.info(
            f"Server retriever queue ready..., Total samples = {len(self.contents)}"
        )
        try:
            while True:
                ## nothing particularly interesting about the request as of right now.
                msg_parts = socket.recv_multipart()
                scout_index, request_number = (int(part.decode()) for part in msg_parts)
                ## update request counter by scout index
                ## synchronizes counters with each client scout
                self.request_counter_by_scout[scout_index] = request_number

                ### response
                sample_path, sample_label = sample_queue.get()

                if sample_path.suffix == ".npy":
                    content = np.load(sample_path)
                else:
                    tmpfile = io.BytesIO()
                    image = Image.open(sample_path).convert("RGB")
                    image.save(tmpfile, format="JPEG", quality=85)
                    content = tmpfile.getvalue()

                socket.send_multipart(
                    [
                        content,
                        sample_label.encode(),
                        str(sample_path).encode(),
                    ]
                )
                served_samples += 1
                if served_samples % 200 == 0:
                    logger.info(
                        f"Server has served {served_samples} / "
                        f"{len(self.contents)} samples..."
                    )

        except Exception as e:
            logger.exception()
            raise e

        ## conditions for scout deployment/redeployment: certain mission time,
        ## after model X has been trained, when <= M scout are alive, 1 or more
        ## scouts have been killed since start, % of mission has passed, reset
        ## and redeploy conditions.
