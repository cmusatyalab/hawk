# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import base64
import binascii
import os
import re
import socket
import time
from pathlib import Path
from typing import TYPE_CHECKING

from logzero import logger

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event
    from typing import Iterator, TextIO

    from ..mission_config import MissionConfig

BOUNDARY_START = "-----BEGIN OPENDIAMOND SCOPECOOKIE-----\n"
BOUNDARY_END = "-----END OPENDIAMOND SCOPECOOKIE-----\n"
COOKIE_VERSION = 1
BASE64_RE = "[A-Za-z0-9+/=\n]+"


def parse_cookie(data: str) -> tuple[str, list[str]]:
    """Parse a (single) scope cookie string and return a ScopeCookie.

    Arguments:
        data {str} -- A single base64-encoded cookie

    Returns:
        [ScopeCookie] -- [description]

    """
    assert isinstance(data, str)

    # Check for boundary markers and remove them
    match = re.match(BOUNDARY_START + "(" + BASE64_RE + ")" + BOUNDARY_END, data)
    if match is None:
        msg = "Invalid boundary markers"
        raise logger.error(msg)
    data = match.group(1)
    # Base64-decode contents
    try:
        data = base64.b64decode(data).decode()
    except TypeError:
        logger.exception("Invalid Base64 data")
        raise
    # Split signature, header, and body
    try:
        signature, data = data.split("\n", 1)
        header, body = data.split("\n\n", 1)
    except ValueError:
        logger.exception("Malformed cookie")
        raise
    # Decode signature
    try:
        binascii.unhexlify(signature)
    except TypeError:
        logger.exception("Malformed signature")
        raise
    # Parse headers
    for line in header.splitlines():
        k, v = line.split(":", 1)
        v = v.strip()
        if k == "Servers":
            servers = [s.strip() for s in re.split("[;,]", v) if s.strip() != ""]
    # Parse body
    scopeurls = [s for s in [u.strip() for u in body.split("\n")] if s != ""]
    scopeurl = "/srv/diamond/INDEXES/GIDIDX" + scopeurls[0].split("/")[-1]
    print(scopeurl)
    logger.info(servers)
    return scopeurl, servers


def define_scope(config: MissionConfig) -> MissionConfig:
    cookie_path = Path.home().joinpath(".hawk", "NEWSCOPE")
    cookie_data = cookie_path.read_text()
    index_path, scouts = parse_cookie(cookie_data)
    config["scouts"] = scouts
    config["dataset"]["index_path"] = index_path
    return config


def get_ip() -> str:
    IP = os.environ.get("HAWK_HOME_ADDRESS")
    if IP is not None:
        return IP

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP


def tailf(fp: TextIO, stop_event: Event | None = None) -> Iterator[str]:
    """Iterate over the lines in the file, wait for more when we hit EOF."""
    fragments: list[str] = []
    while stop_event is None or not stop_event.is_set():
        for line in fp:
            # this is only an optimization to avoid maintaining and
            # concatenating the list of fragments
            if len(fragments) == 0 and line[-1] == "\n":
                yield line
                continue

            fragments.append(line)
            if line[-1] == "\n":
                yield "".join(fragments)
                fragments = []

        # got to EOF, wait for more.
        time.sleep(0.5)
