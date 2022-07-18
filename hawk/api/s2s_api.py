# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import zmq
from logzero import logger

from hawk import api
from hawk.proto.messages_pb2 import * 

def s2s_receive_request(s2s_input, s2s_output):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f'tcp://0.0.0.0:{api.S2S_PORT}')
    try:
        while True:
            msg = socket.recv_pyobj()
            method = msg.get('method')
            req = msg.get('msg')
            logger.info("Received S2S call")
            s2s_input.put((method, req))
            reply = s2s_output.get()
            socket.send(reply)
    except Exception as e:
        logger.exception(e)
        raise e

class S2SServicer(object):

    def __init__(self, mission):
        self._mission = mission

    def s2s_get_tile(self, msg):
        """To fetch contents of requested tile ids
        Call made by coordinator to (parent) scout where tile is present 
        The tile is labeled if labels present
        
        Input: TileMetadata
        Output: HawkObject
        """
        try:    
            request = LabelWrapper()
            request.ParseFromString(msg)
            label = request
            # Fetch data from dataretriever
            object = self._mission.retriever.get_object(
                object_id=label.objectId 
            )
            # Assuming data requirement in Distribute positives
            if label.imageLabel != '0':
                # Transmit data to coordinator
                response = object.SerializeToString()
                logger.info("Fetch Tile for id {} parent {} Reply {}".format(
                    label.objectId, label.scoutIndex, len(response)))
            else:
                response = api.Empty
            # Store labels
            self._mission.store_labeled_tile(
                LabeledTile(
                    object=object,
                    label=label,
                ) 
            )
        except Exception as e:
            logger.exception(e)
            raise e
        finally:
            return response
    
    def s2s_add_tile_and_label(self, msg):
        """To transmit tile content and labels
        Call made by coordinator to non-parent scouts 
        Saves tile content and label locally
        
        Input: LabeledTile
        Output: None 
        """
        try:
            request = LabeledTile()
            request.ParseFromString(msg)

            self._mission.store_labeled_tile(request)
            
            return 
        except Exception as e:
            logger.exception(e)
            raise e
        finally:
            return api.Empty 
    
    def s2s_send_data(self):
        """To send internal messages (bytes) between scouts
        """
        pass
