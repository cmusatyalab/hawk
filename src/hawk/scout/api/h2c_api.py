# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Home to Scouts internal api calls
"""

from multiprocessing.connection import _ConnectionBase

import zmq
from logzero import logger

from ...proto.messages_pb2 import SendLabels


class H2CSubscriber:
    @staticmethod
    def h2c_receive_labels(label_conn: _ConnectionBase, h2c_port: int) -> None:
        """API call to receives labels from HOME

        Uses ZeroMQ PUSH/PULL protocol for async label transfer
        Network is bandwidth constricted using FireQOS.

        Args:
            label_conn: mp.Pipe connection object

        Returns:
            str: serializes LabelWrapper message
        """
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.bind(f"tcp://*:{h2c_port}")
        try:
            while True:
                msg = socket.recv()
                resp = SendLabels()
                resp.ParseFromString(msg)
                label = resp.label.SerializeToString()
                label_conn.send(label)
        except Exception as e:
            logger.exception(e)
            raise e
