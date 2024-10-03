# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import argparse
import logging
import signal
import sys
import threading
import traceback
from typing import Any

import logzero
import multiprocessing_logging
import torchvision
import zmq
from logzero import logger
from prometheus_client import start_http_server as start_metrics_server

from ..ports import A2S_PORT
from .api.a2s_api import A2SAPI
from .core.utils import log_exceptions

logzero.loglevel(logging.INFO)
torchvision.set_image_backend("accimage")


def handler_signals(signum: int, _frame: Any) -> None:
    sys.exit(0)


def dumpstacks(_: Any, __: Any) -> None:
    traceback.print_stack()
    id2name = {th.ident: th.name for th in threading.enumerate()}
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append("\n# Thread: %s(%d)" % (id2name.get(threadId, ""), threadId))
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
            if line:
                code.append("  %s" % (line.strip()))
    print("\n".join(code))


@log_exceptions
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2s-port", type=int, default=A2S_PORT)
    parser.add_argument("--metrics-port", type=int)
    args = parser.parse_args()

    metrics_port = args.a2s_port + 3 if args.metrics_port is None else args.metrics_port
    start_metrics_server(port=metrics_port)

    multiprocessing_logging.install_mp_handler()

    learning_module_api = A2SAPI(args.a2s_port)
    a2s_methods = {
        k.encode("utf-8"): getattr(learning_module_api, k)
        for k in dir(learning_module_api)
        if callable(getattr(learning_module_api, k)) and k.startswith("a2s_")
    }

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{args.a2s_port}")
    logger.info("Starting Hawk server")
    try:
        while True:
            method, req = socket.recv_multipart()
            logger.info(f"Received A2S call {method.decode()} {len(req)}")
            reply = a2s_methods[method](req)
            socket.send(reply)
            if method == b"a2s_stop_mission":
                break
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handler_signals)
    signal.signal(signal.SIGTERM, handler_signals)
    main()
