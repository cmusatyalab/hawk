# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Home to Scouts internal api calls."""

from multiprocessing.connection import _ConnectionBase

import zmq
from logzero import logger


class H2CSubscriber:
    @staticmethod
    def h2c_receive_labels(label_conn: _ConnectionBase, h2c_port: int) -> None:
        """API call to receives labels from HOME.

        Uses ZeroMQ PUSH/PULL protocol for async label transfer
        Network is bandwidth constricted using FireQOS.

        Args:
            label_conn: mp.Pipe connection object

        """
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.bind(f"tcp://*:{h2c_port}")
        try:
            while True:
                msg = socket.recv()
                label_conn.send(msg)
        except Exception as e:
            logger.exception(e)
            raise
