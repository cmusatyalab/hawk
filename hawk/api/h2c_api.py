# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import zmq
from logzero import logger

from hawk import api
from hawk.proto.messages_pb2 import SendLabels

    
class H2CSubscriber(object):
    @staticmethod
    def h2c_receive_labels(host_ip, label_conn):
        """Received labels from home 
        Subscribe: SendLabels
        """
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.bind("tcp://*:{}".format(api.H2C_PORT))
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
        
