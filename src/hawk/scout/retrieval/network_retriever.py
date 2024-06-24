# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import io
import math
import time
from collections import defaultdict
from pathlib import Path
import queue

import numpy as np
from logzero import logger
from PIL import Image
import zmq

from ...proto.messages_pb2 import FileDataset
from ...proto.messages_pb2 import NetworkDataset
from ..core.attribute_provider import HawkAttributeProvider
from ..core.object_provider import ObjectProvider
from ..stats import collect_metrics_total
from .retriever import Retriever


class NetworkRetriever(Retriever):
    def __init__(self, mission_id: str, dataset: NetworkDataset, host_name: str):
        super().__init__(mission_id)
        self.network = True
        self.this_host_name = host_name.split(":")[0]
        self._dataset = dataset
        self._network_server_host = dataset.dataServerAddr ## pulled from home config
        self._network_server_port = dataset.dataServerPort ## 6105 ## pulled from home config
        self._timeout = dataset.timeout ## from home config default 20
        self._resize = dataset.resizeTile
        logger.info("In NETWORK RETRIEVER INIT...")

        index_file = Path(self._dataset.dataPath)
        self._data_root = index_file.parent.parent
        self.contents = index_file.read_text().splitlines()
        self.total_tiles = len(self.contents)
        ## for network retriever, will update total_tiles to equal tiles processes as clients unaware of total tiles per scout a priori

        num_tiles = self._dataset.numTiles
        key_len = math.ceil(self.total_tiles / num_tiles)

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
        self.total_objects.set(self.total_tiles)

        self.request_counter_by_scout = {}
        self.sample_count = 0

    def stream_objects(self) -> None:
        super().stream_objects()
        assert self._context is not None
        context = zmq.Context()
        client_socket = context.socket(zmq.REQ)
        client_socket.connect(f"tcp://{self._network_server_host}:{self._network_server_port}")  ## setup socket

        ## next will need to manage some flags for when network sample retrieval stop and starts, otherwise the net retriever should continuously send requests to the server.
        scout_index = str(self.scout_index).encode('utf-8')
        time_start = time.time()
        while True: ## need additional conditions here to break out of this loop, or to block at some line before retrieving the next sample.
            ## prepare and send request
            self.sample_count += 1
            self.total_tiles = self.sample_count ## next step will need the server to send the number of remaining samples on server, which should decrease over time.
            msg = [scout_index, str(self.sample_count).encode('utf-8')]
            client_socket.send_multipart(msg)
            
            ### receive and process
            content, label, path = client_socket.recv_multipart()
            label, path = label.decode(), path.decode()
            object_id = f"/{label}/collection/id/" + path
            attributes = self.set_tile_attributes(object_id, label)

            self.put_objects(
                ObjectProvider(
                    object_id,
                    content,
                    HawkAttributeProvider(attributes, Path(path), self._resize),
                    int(label),
                )
            )

            if self.sample_count % self._dataset.numTiles == 0:
                retrieved_tiles = collect_metrics_total(self.retrieved_objects)
                logger.info(f"{retrieved_tiles} / {self.total_tiles} RETRIEVED")
                time_passed = time.time() - time_start
                if time_passed < self._timeout:
                    logger.info(f"About to sleep at: {time.time()}")
                    time.sleep(self._timeout - time_passed)
            time_start = time.time()
            ## may need to adjust this timeout of 20 to accomodate higher retrieval rates.

    def server(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://0.0.0.0:{self._network_server_port}")  ## setup socket
        logger.info(f"Server socket ready...")
        sample_queue = queue.Queue()
        for line in self.contents:
            path, label = line.split()
            sample_queue.put((Path(path), label))
        logger.info(f"Server retriever queue ready..., Total samples = {len(self.contents)}")
        try:
            while True:
                ## nothing particularly interesting about the request as of right now.
                msg_parts = socket.recv_multipart()
                scout_index, request_number = [part.decode('utf-8') for part in msg_parts]
                ## update request counter by scout index
                self.request_counter_by_scout[scout_index] = request_number ## synchronizes counters with each client scout

                ### response
                sample_path, sample_label = sample_queue.get()
                if sample_path.suffix == ".npy":
                    content = np.load(sample_path)
                else:
                    tmpfile = io.BytesIO()
                    image = Image.open(sample_path).convert("RGB")
                    image.save(tmpfile, format="JPEG", quality=85)
                    content = tmpfile.getvalue()
                socket.send_multipart([content, sample_label.encode('utf-8'), str(sample_path).encode('utf-8')])

        except Exception as e:
            logger.exception()
            raise e
        
        ## conditions for scout deployment/redeployment: certain mission time, after model X has been trained, when <= M scout are alive, 1 or more scouts have been killed since start, % of mission has passed, reset and redeploy conditions.