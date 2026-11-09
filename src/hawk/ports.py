# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

import os

# HAWK_PORT_OFFSET shifts the default ports, to run more than one home and
# scout pair on a host
_OFFSET = int(os.environ.get("HAWK_PORT_OFFSET", "0"))

A2S_PORT = 6100 + _OFFSET
S2H_PORT = 6177 + _OFFSET
H2C_PORT = 6101 + _OFFSET
H2A_PORT = 6200 + _OFFSET
S2S_PORT = 6102 + _OFFSET
NET_RETRIEVER_PORT = 6105 + _OFFSET

HOME_METRICS_PORT = 6178 + _OFFSET
# SCOUT_METRICS_PORT = A2S_PORT + 3  # 6003
