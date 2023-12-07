# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Scout to Home internal api calls
"""

from multiprocessing.connection import _ConnectionBase

import zmq
from logzero import logger

from ...ports import S2H_PORT


class S2HPublisher:
    @staticmethod
    def s2h_send_tiles(home_ip: str, result_conn: _ConnectionBase) -> None:
        """API call to send results to HOME

        Uses ZeroMQ PUSH/PULL protocol for async result transfer
        Network is bandwidth constricted using FireQOS.

        Args:
            home_ip: IP address of HOME
            result_conn: mp.Pipe connection object

        Returns:
            str: transmits serialized SendTiles message
        """
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.connect(f"tcp://{home_ip}:{S2H_PORT}")

        try:
            while True:
                msg = result_conn.recv()
                if not len(msg):
                    logger.debug("[SendTile]: GOT RESULT NONE")
                    return
                socket.send(msg)
        except Exception as e:
            logger.exception(e)
            raise e
