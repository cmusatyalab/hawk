# SPDX-FileCopyrightText: 2022,2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import socket

import zmq

from ...ports import S2S_PORT


class HawkStub:
    def __init__(self, host: str, this_host: str) -> None:
        self.hostname, port = (host.rsplit(":", 1) + [str(S2S_PORT)])[:2]
        if host == this_host:
            # open server connection
            self.internal = None
        else:
            ip = socket.gethostbyname(self.hostname)
            self.zmq_context = zmq.Context()
            self.internal = self.zmq_context.socket(zmq.REQ)
            self.internal.connect(f"tcp://{ip}:{port}")
