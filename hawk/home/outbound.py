# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Home Outbound process: sending labels from HOME to SCOUTS 
"""

import json
import queue
import socket

import zmq
from logzero import logger

from ..proto import Empty
from ..proto.messages_pb2 import LabelWrapper, SendLabels


class OutboundProcess:
    def __init__(self, train_location: str = "scout") -> None:
        """Outbound messages from HOME to SCOUT

        API calls from home to scouts to to send tile labels from home to coordinator scout. 
        Uses PUSH/PULL messaging protocol. The network is bandwidth constricted using FireQos.
        """
        train_location = train_location.lower()
       
        if train_location == "scout":
            self.transmit_func = self.scout_send_labels
        else:
            raise NotImplementedError(f"Training Location {train_location} Not Implemented")

    def send_labels(self, scout_ips, h2c_port, result_q, stop_event):
        """API call to send messages from Home to Scouts"""

        self.scout_ips = scout_ips
        self.result_q = result_q
        self.stop_event = stop_event
        self.zmq_context = zmq.Context()
        try:
            self.stubs = {}
            for index, ip in enumerate(self.scout_ips):
                h2c_socket = self.zmq_context.socket(zmq.PUSH)
                h2c_socket.connect(f"tcp://{ip}:{h2c_port}")
                self.stubs[index] = h2c_socket

            self.transmit_func()

        except KeyboardInterrupt as e:
            raise e 
    
    def scout_send_labels(self):
        """Function to serialize labels for transmission"""
        msg = Empty
        scout_index = None
        try:
            while not self.stop_event.is_set():
                try:
                    label_path = self.result_q.get()
                except queue.Empty:
                    continue
                
                # Read meta data and create LabelWrapper
                data = {}
                with open(label_path, "r") as f:
                    data = json.load(f)

                label=LabelWrapper(
                    objectId = data['objectId'],
                    scoutIndex = data['scoutIndex'],
                    imageLabel = data['imageLabel'],
                    boundingBoxes = data['boundingBoxes']
                )
                response = SendLabels(
                    label=label,
                )
                msg = response.SerializeToString()
                logger.info("Send labels {} {} {} {}".format(
                    data['imageLabel'], data['scoutIndex'], data['objectId'], len(msg)))
                scout_index = int(data['scoutIndex'])
                self.stubs[scout_index].send(msg)

        except (IOError, KeyboardInterrupt) as e:
            logger.error(e)

        return msg, [scout_index]
