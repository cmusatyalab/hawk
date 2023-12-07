# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Scout to Scout internal api calls
"""

from multiprocessing import Queue
from typing import TYPE_CHECKING, Tuple

import zmq
from logzero import logger

from ...ports import S2S_PORT
from ...proto import Empty
from ...proto.messages_pb2 import LabeledTile, LabelWrapper

if TYPE_CHECKING:
    from ..core.mission import Mission


def s2s_receive_request(
    s2s_input: Queue[Tuple[bytes, bytes]], s2s_output: Queue[bytes]
) -> None:
    """Function to receive and invoke S2S api calls

    Uses Request-Response messaging protocol

    Args:
        s2s_input: mp.Queue containing requests
        s2s_output: mp.Queue containing responses

    Returns:
        str: serialized output responses
    """
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{S2S_PORT}")
    try:
        while True:
            method, req = socket.recv_multipart()
            logger.info("Received S2S call")
            s2s_input.put((method, req))
            reply = s2s_output.get()
            socket.send(reply)
    except Exception as e:
        logger.exception()
        raise e


class S2SServicer:
    def __init__(self, mission: Mission):
        self._mission = mission

    def s2s_get_tile(self, msg: bytes) -> bytes:
        """API call to fetch contents of requested tile ids

        Call made by COORDINATOR to (PARENT) scout where image is present

        Args:
            msg: serialized LabelWrapper message

        Returns:
            str: transmits serialized HawkObject message
        """
        try:
            request = LabelWrapper()
            request.ParseFromString(msg)
            label = request
            # Fetch data from dataretriever
            obj = self._mission.retriever.get_object(object_id=label.objectId)
            # Assuming data requirement in Distribute positives
            if label.imageLabel != "0":
                # Transmit data to coordinator
                response = obj.SerializeToString()
                logger.info(
                    "Fetch Tile for id {} parent {} Reply {}".format(
                        label.objectId, label.scoutIndex, len(response)
                    )
                )
            else:
                response = Empty
            # Store labels
            self._mission.store_labeled_tile(LabeledTile(obj=obj, label=label))
        except Exception as e:
            logger.exception(e)
            response = Empty
        return response

    def s2s_add_tile_and_label(self, msg: bytes) -> bytes:
        """API call to add tile content and labels

        Call made by COORDINATOR to non-PARENT scouts

        Args:
            msg: serialized LabeledTile message
        """
        try:
            request = LabeledTile()
            request.ParseFromString(msg)

            self._mission.store_labeled_tile(request)
        except Exception as e:
            logger.exception(e)
        return Empty
