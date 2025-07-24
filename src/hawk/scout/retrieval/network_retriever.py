# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Iterator

import zmq
from logzero import logger

from ...objectid import ObjectId
from .random_retriever import RandomRetriever, RandomRetrieverConfig


class BalanceMode(Enum):
    LOCALLY_CONSTANT = "locally_constant"
    GLOBALLY_CONSTANT = "globally_constant"


class NetworkRetrieverConfig(RandomRetrieverConfig):
    server_host: str
    server_port: int
    balance_mode: BalanceMode = BalanceMode.LOCALLY_CONSTANT


class NetworkRetriever(RandomRetriever):
    config_class = NetworkRetrieverConfig
    config: NetworkRetrieverConfig

    def __init__(self, config: NetworkRetrieverConfig) -> None:
        super().__init__(config)

        # avoid double counting by only allowing server to report how many
        # samples have been processed. (fix this when we launch the server thread)
        self.total_objects.set(0)

        self.globally_constant = (
            self.config.balance_mode == BalanceMode.GLOBALLY_CONSTANT
        )

        # True unless local scout deploy conditions prevent
        self.scml_active_condition = True

        logger.info("In NETWORK RETRIEVER INIT...")
        self.request_counter_by_scout: dict[int, int] = {}
        self.sample_count = 0

    def _run_threads(self) -> None:
        assert self._context is not None
        scout_index = self._context.scout_index
        scout = self._context.scouts[scout_index]

        if self.config.server_host != scout.hostname:
            logger.info("Starting server thread in retriever...")
            threading.Thread(target=self.server, name="network_server").start()
        else:
            super()._run_threads()

    def get_next_objectid(self) -> Iterator[ObjectId | None]:
        assert self._context is not None

        # network retriever remains idle until we get the network connection
        # up and the scml conditions are satisfied.
        self.current_deployment_mode = "Idle"

        context = zmq.Context()
        client_socket = context.socket(zmq.REQ)
        client_socket.connect(
            f"tcp://{self.config.server_host}:{self.config.server_port}",
        )  ## setup socket

        ## next will need to manage some flags for when network sample
        ## retrieval stop and starts, otherwise the net retriever should
        ## continuously send requests to the server.
        scout_index = str(self._context.scout_index).encode()
        logger.info(f"SCML options from mission: {self._context.scml_deploy_options}")
        while True:
            ## if sent deployment order from home or preset conditions are reached.
            if self.scml_active_mode is not None:
                active = self.scml_active_mode
            else:
                mission_time = time.time() - self._start_time
                model_version = self._context.model_version
                scml_deploy_options = self._context.scml_deploy_options

                # when start conditions are specified we should only switch to
                # active state when any of the start conditions are triggered.
                # Otherwise we assume we begin from an active state that may
                # get disabled by any not_before_ or end_ conditions.
                start_conditions = {"start_time", "start_on_model"}
                active = not bool(start_conditions.intersection(scml_deploy_options))

                if "start_time" in scml_deploy_options:
                    active |= mission_time >= scml_deploy_options["start_time"]

                if "start_on_model" in scml_deploy_options:
                    active |= model_version >= scml_deploy_options["start_on_model"]

                if "not_before_time" in scml_deploy_options:
                    active &= mission_time >= scml_deploy_options["not_before_time"]
                if "not_before_model" in scml_deploy_options:
                    active &= model_version >= scml_deploy_options["not_before_model"]

                if "end_time" in scml_deploy_options:
                    active &= mission_time < scml_deploy_options["end_time"]
                if "end_on_model" in scml_deploy_options:
                    active &= model_version < scml_deploy_options["end_on_model"]

            self.current_deployment_mode = "Active" if active else "Idle"

            ## i.e. if it's Dead or Idle, 'Dead' not fully implemented yet
            if not active:
                logger.info(
                    "Too early to start retrieving or in idle mode... "
                    f"{time.time() - self._start_time}",
                )
                time.sleep(5)
                continue
            ## need additional conditions here to break out of this loop, or to
            ## block at some line before retrieving the next sample.
            ## prepare and send request
            self.sample_count += 1
            self.total_tiles = self.sample_count
            ## next step will need the server to send the number of remaining
            ## samples on server, which should decrease over time.
            msg = [scout_index, str(self.sample_count).encode()]
            client_socket.send_multipart(msg)

            ### receive and process
            oid = client_socket.recv_multipart()[0]
            yield ObjectId(oid.decode())

            if self.sample_count % self.config.tiles_per_frame == 0:
                # It does not matter if we're globally or locally constant.
                # If it was global, we should be throttled enough in the RPC
                # that we never have to spend extra time sleeping here.
                yield None

    def server(self) -> None:
        self.total_objects.set(self.total_tiles)
        self.current_deployment_mode = "Server"
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://0.0.0.0:{self.config.server_port}")  ## setup socket
        logger.info("Server socket ready...")
        logger.info(
            f"Server retriever queue ready..., Total samples = {self.total_tiles}",
        )
        try:
            time_start = time.time()
            for ntiles, object_id in enumerate(super().get_next_objectid(), 1):
                if object_id is None:
                    continue

                ## nothing particularly interesting about the request as of right now.
                msg_parts = socket.recv_multipart()
                scout_index, request_number = (int(part.decode()) for part in msg_parts)

                ## update request counter by scout index
                ## synchronizes counters with each client scout
                self.request_counter_by_scout[scout_index] = request_number

                ### response
                socket.send_multipart([object_id.serialize_oid()])

                if ntiles % self.config.tiles_per_frame != 0:
                    continue

                # we've sent a frame worth of oids, check if we need to slow down.
                logger.info(
                    f"Server has served {ntiles} / {self.total_tiles} samples...",
                )

                if not self.globally_constant:
                    continue

                # using a globally constant rate? we sleep at the server!
                time_passed = time.time() - time_start
                if time_passed < self.config.timeout:
                    logger.info(f"About to sleep at: {time.time()}")
                    time.sleep(self.config.timeout - time_passed)
                time_start = time.time()

        except Exception:
            logger.exception()
            raise

        ## conditions for scout deployment/redeployment: certain mission time,
        ## after model X has been trained, when <= M scout are alive, 1 or more
        ## scouts have been killed since start, % of mission has passed, reset
        ## and redeploy conditions.
