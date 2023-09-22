# SPDX-FileCopyrightText: 2022,2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import socket

import zmq

from ...ports import S2S_PORT


class HawkStub:

    def __init__(self, host, this_host):

        if host == this_host:
            # open server connection
            self.internal = None
        else:
            hostname, port = (host.rsplit(':', 1) + [S2S_PORT])[:2]
            ip = socket.gethostbyname(hostname)
            self.zmq_context = zmq.Context()
            self.internal = self.zmq_context.socket(zmq.REQ)
            self.internal.connect(f"tcp://{ip}:{port}")
