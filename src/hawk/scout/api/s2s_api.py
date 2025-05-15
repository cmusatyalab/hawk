# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Scout to Scout internal api calls"""

from __future__ import annotations

from multiprocessing import Queue
from typing import TYPE_CHECKING

import zmq
from logzero import logger

from ...objectid import ObjectId
from ...proto import Empty
from ...proto.messages_pb2 import LabeledTile, SendLabel

if TYPE_CHECKING:
    from ..core.mission import Mission


def s2s_receive_request(
    s2s_input: Queue[tuple[bytes, bytes]], s2s_output: Queue[bytes], s2s_port: int
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
    socket.bind(f"tcp://0.0.0.0:{s2s_port}")
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
            msg: serialized SendLabel message

        Returns:
            str: transmits serialized HawkObject message
        """
        try:
            label = SendLabel()
            label.ParseFromString(msg)
            # Fetch data from dataretriever
            objectId = ObjectId(label._objectId)
            obj = self._mission.retriever.read_object(objectId)
            assert obj is not None
            # Assuming data requirement in Distribute positives
            if label.boundingBoxes:
                # Transmit data to coordinator
                response = obj.SerializeToString()
                logger.info(
                    f"Fetch Tile for {objectId} parent {label.scoutIndex}"
                    f" Reply {len(response)}"
                )
            else:
                response = Empty

            # Store labels (positives and negatives since we're the originating scout)
            labeled_tile = LabeledTile(obj=obj, boundingBoxes=label.boundingBoxes)
            self._mission.store_labeled_tile(labeled_tile)
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

            # if mode is active dont call this for scml
            self._mission.store_labeled_tile(request, net=True)
        except Exception as e:
            logger.exception(e)
        return Empty
