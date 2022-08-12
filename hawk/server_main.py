# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import logging
import os
import signal
import sys
import threading
import traceback

import logzero
import multiprocessing_logging
import torchvision
import zmq
from logzero import logger

from hawk.api import A2S_PORT
from hawk.api.a2s_api import A2SAPI 
from hawk.core.utils import log_exceptions

logzero.loglevel(logging.INFO)
torchvision.set_image_backend('accimage')


def dumpstacks(_, __):
    traceback.print_stack()
    id2name = dict([(th.ident, th.name) for th in threading.enumerate()])
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append("\n# Thread: %s(%d)" % (id2name.get(threadId, ""), threadId))
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
            if line:
                code.append("  %s" % (line.strip()))
    print("\n".join(code))


@log_exceptions
def main():
    multiprocessing_logging.install_mp_handler()

    learning_module_api = A2SAPI(A2S_PORT)
    a2s_methods = dict((k, getattr(learning_module_api, k))
        for k in dir(learning_module_api)
        if callable(getattr(learning_module_api, k)) and
        not k.startswith('_'))

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f'tcp://0.0.0.0:{A2S_PORT}')
    logger.info("Starting Hawk server")
    try:
        while True:
            msg = socket.recv_pyobj()
            method = msg.get('method')
            req = msg.get('msg')
            logger.info("Received A2S call {} {}".format(method, len(req)))
            if len(req):
                reply = a2s_methods[method](req)
            else:
                reply = a2s_methods[method]()
            socket.send(reply)
    except KeyboardInterrupt:
        pass 
    except Exception as e:
        logger.exception(e)
        raise e
    finally:
        pid = os.getpid()
        os.kill(pid, signal.SIGKILL)

if __name__ == '__main__':
    main()
