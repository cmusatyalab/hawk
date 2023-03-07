# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

"""Scout to Home internal api calls
"""

import zmq
from logzero import logger

from hawk import api 

class S2HPublisher(object):
    @staticmethod
    def s2h_send_tiles(home_ip, result_conn): 
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
        socket.connect("tcp://{}:{}".format(home_ip, api.S2H_PORT))

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
