# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

"""Home to Scouts internal api calls
"""

import zmq
from logzero import logger

from ...ports import H2C_PORT
from ...proto.messages_pb2 import SendLabels


class H2CSubscriber(object):
    @staticmethod
    def h2c_receive_labels(label_conn):
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
        socket.bind(f"tcp://*:{H2C_PORT}")
        try:
            while True:
                msg = socket.recv()
                resp = SendLabels()
                resp.ParseFromString(msg)
                msg = resp.label
                label_conn.send(msg.SerializeToString())
        except Exception as e:
            logger.exception(e)
            raise e
        
