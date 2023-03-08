# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

"""Home Inbound process: receiving result from SCOUTS
"""

import json
import zmq

from home import ZFILL
from logzero import logger
from pathlib import Path 

from hawk.api import S2H_PORT
from hawk.proto.messages_pb2 import SendTiles


class InboundProcess:
    """Inbound messages to HOME from SCOUT

    API calls from home to scouts to to send tile labels from home to coordinator scout. 
    Uses PUSH/PULL messaging protocol. The network is bandwidth constricted using FireQos.
    """
    def __init__(self, 
                 tile_dir: Path, 
                 meta_dir: Path,
                 train_location: str = "scout") -> None:
        self._tile_dir = tile_dir
        self._meta_dir = meta_dir
        
        self._running = False 
        
        train_location = train_location.lower()
        
        self._save_attribute = "tile.jpeg"
        if train_location == "scout":
            self._save_attribute = "thumbnail.jpeg"
        else:
            raise NotImplementedError(f"Training Location {train_location} Not Implemented")
    
        self._count = 1
    
    def receive_data(self, result_q, stop_event):
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.bind('tcp://*:{}'.format(S2H_PORT))
        logger.info("Inbound Receive data started")    
        try:
            while not stop_event.is_set():
                msg = socket.recv()
                request = SendTiles()
                request.ParseFromString(msg)
                object_id = request.objectId
                parent_scout = request.scoutIndex
                score = request.score
                data_name = (str(self._count)).zfill(ZFILL)

                data = request.attributes
                byte_size = request.ByteSize()

                assert self._save_attribute in data

                # Tiles and thumbnails are jpegs

                with open(self._tile_dir/ f"{data_name}.jpeg", "wb") as f:
                    img_array = data[self._save_attribute]
                    f.write(bytearray(img_array))  

                meta_data = {
                    'objectId': object_id,
                    'scoutIndex': parent_scout,
                    'score': score,
                    'size': byte_size, 
                } 
                meta_path = self._meta_dir/ f"{data_name}.json"

                with open(meta_path, "w") as f:
                    json.dump(meta_data, f) 

                logger.info(f"Received {object_id} {parent_scout} {score}") 
                logger.info("SAVING TILES {}".format(self._tile_dir/ f"{data_name}.jpeg"))
                result_q.put(meta_path)
                self._count += 1

        except (Exception, IOError, KeyboardInterrupt) as e:
            logger.error(e)
                        