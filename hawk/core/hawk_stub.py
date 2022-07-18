# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import zmq
import socket 

class HawkStub(object):

    def __init__(self, ip, port, host_ip):
        
        if ip == host_ip:
            # open server connection 
            self.internal = None
        else:
            domain_name = ip 
            ip = socket.gethostbyname(domain_name)
            context = zmq.Context()
            client = context.socket(zmq.REQ)
            self.internal = client
            self.internal.connect("tcp://{}:{}".format(ip, port))
