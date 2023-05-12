# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

# Usage: python home_flutter.py 

import base64
import io
import json
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import yaml
import zmq

from admin import Admin 
from datetime import datetime
from flask import Flask, request
from hawk.core.utils import get_ip
from hawk.api import  H2A_PORT
from home import *
from inbound import InboundProcess
from logzero import logger
from outbound import OutboundProcess
from pathlib import Path
from PIL import Image
from script_labeler import ScriptLabeler
from typing import Iterable
from ui_labeler import UILabeler
from utils import define_scope, write_config

REMOTE_USER = 'root'
CONFIG = os.getenv('HOME')+"/.hawk/config.yml" # PUT this in .hawk
home_path = os.path.dirname(os.path.realpath(__file__))
app_data = {}
app = Flask(__name__)


def restart_scouts(scouts: Iterable[str]):
    host_file = os.path.join(home_path, 'hosts')
    restart_file = os.path.join(home_path, 'scout-restart.sh')
    with open(host_file, 'w') as f:
        f.write("\n".join(scouts))
    scout_startup_cmd = f"parallel-ssh -t 0 -h {host_file} \
        -l {REMOTE_USER} -P -I<{restart_file} > /dev/null"
    os.system(scout_startup_cmd)
    logger.info("Restarting scouts")
    return


def stop_mission():
    global app_data

    if 'started' in app_data:
        logger.info("Stopping Mission")
        home_admin = app_data['home-admin']
        home_admin.stop_mission()

        restart_scouts(app_data['scouts'])

        # Stopping processes
        processes = app_data['processes']
        for p in processes:
            p.terminate()

    app_data = {}
    return 


def handler_signals(signum, frame):
    stop_mission()
    sys.exit(0)


def get_results():
    image_root = os.path.dirname(app_data['image-dir'])
    #result_cmd = f"python {home_path}/result_stream.py {image_root} > /dev/null"
    result_cmd = f"python {home_path}/result_stream.py {image_root}"
    logger.info(result_cmd)
    os.system(result_cmd)
    return


def configure_mission(filter_config):
    # Stop previous missions
    stop_mission()
    
    config_path = CONFIG 
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Setting up mission 
    mission_name = config.get('mission-name', "test")
    
    assert (Path(config['train_strategy'].get('initial_model_path', '')).exists() or 
            Path(config['train_strategy'].get('bootstrap_path', '')).exists() or 
            Path(config['train_strategy'].get('example_path', '')).exists())    

    mission_id = "_".join([mission_name, 
                               datetime.now().strftime('%Y%m%d-%H%M%S')])
    
    if config['dataset']['type'] == 'cookie':
        logger.info("Reading Scope Cookie")
        logger.info(f"Participating scouts \n{config['scouts']}")
        config = define_scope(config)

   
    bandwidth = config.get('bandwidth', "100")
    assert int(bandwidth) in [100, 30, 12], "Fireqos script may not exist for {}".format(bandwidth)
    config['bandwidth'] = ["[[-1, \"{}k\"]]".format(bandwidth) for _ in config['scouts']] 
    
    config['train_strategy']['type'] = filter_config['name']

    # Add more filters here 
    if filter_config['name'] == "fsl":
        support_string = filter_config['args']['support']
        image = Image.open(io.BytesIO(base64.b64decode(support_string)))
        # Resize to 256 x 256
        dim = (256, 256)
        image = image.resize(dim, Image.LANCZOS)
        image_path = "/home/shilpag/.hawk/example.jpg"
        image.save(image_path)
        config['train_strategy']['example_path'] = image_path
    else:
        raise NotImplementedError("Unknown train_strategy {}".format(filter_config['name']))

    # create local directories 
    mission_dir = Path(config['home-params']['mission_dir'])
    mission_dir = mission_dir / mission_id
    logger.info(mission_dir)

    log_dir = mission_dir / 'logs'
    config['home-params']['log_dir'] = str(log_dir)
    end_file = log_dir / 'end'

    image_dir = mission_dir / 'images'
    meta_dir = mission_dir / 'meta'
    label_dir = mission_dir / 'labels'
    
    app_data['image-dir'] = image_dir
    app_data['meta-dir'] = meta_dir
    app_data['label-dir'] = label_dir

    log_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    # Save config file to log_dir
    config_path = log_dir / 'hawk.yml'
    write_config(config, config_path)

    # Setting up helpers
    scout_ips = config['scouts']
    app_data['scouts'] = config['scouts']
    restart_scouts(scout_ips)
    processes = [] 
    stop_event = mp.Event()
    meta_q = mp.Queue()
    label_q = mp.Queue()
    stats_q = mp.Queue()
    stats_q.put((0,0,0))

    thread = threading.Thread(target = get_results)
    thread.start()
    
    try:

        # Starting home to admin conn 
        context = zmq.Context()
        h2a_socket = context.socket(zmq.REQ)
        h2a_socket.connect(f'tcp://127.0.0.1:{H2A_PORT}')

        # Setup admin 
        home_ip = get_ip()

        logger.info("Starting Admin Process")
        home_admin = Admin(home_ip, mission_id) # , explicit_start=True)
        p = mp.Process(target=home_admin.receive_from_home, kwargs={'stop_event': stop_event,
                                                                    'stats_q': stats_q})
        app_data['home-admin'] = home_admin
        processes.append(p)
        p.start()
    
        # Start inbound process
        logger.info("Starting Inbound Process")
        home_inbound = InboundProcess(image_dir, 
                                      meta_dir, 
                                      config)
        p = mp.Process(target=home_inbound.receive_data, kwargs={'result_q': meta_q,
                                                                 'stop_event': stop_event})
        processes.append(p)
        p.start()
        
        # start outbound process  
        logger.info("Starting Outbound Process")
        home_outbound = OutboundProcess()
        p = mp.Process(target=home_outbound.send_labels, kwargs={'scout_ips': scout_ips,
                                                                 'result_q': label_q,
                                                                 'stop_event': stop_event})
        p.start()
        processes.append(p)

        app_data['processes'] = processes

        # Send config file to admin 
        # send msg "<config> <path to config file>"
        
        h2a_socket.send_string(f"config {config_path}")
        h2a_socket.recv()
   
    except KeyboardInterrupt as e:
        logger.error(e)


@app.route('/start', methods = ['POST'])
def startMission():

    request_data = request.data #getting the response data
    request_data = json.loads(request_data.decode('utf-8'))
    name = request_data['name'] 
    response = f'{name}' 
    logger.info(response)
    configure_mission(request_data)
    app_data['started'] = True
    return ('', 204)


# @app.route('/start', methods = ['POST'])
# def startMission():

#     request_data = request.data #getting the response data
#     request_data = json.loads(request_data.decode('utf-8')) #converting it from json to key value pair
#     name = request_data['name'] #assigning it to name
#     response = f'{name}' #re-assigning response with the name we got from the user
#     # logger.info(response)
#     if 'home-admin' in app_data:
#         logger.info("Starting Mission")
#         home_admin = app_data['home-admin']
#         home_admin.start_mission()
#         app_data['started'] = True
#     else:
#         logger.info("MISSION NOT CONFIGURED")
#     return ('', 204)


@app.route('/stop', methods = ['POST'])
def stopMission():

    request_data = request.data 
    request_data = json.loads(request_data.decode('utf-8')) 
    name = request_data['name'] 
    response = f'{name}' 
    if 'home-admin' in app_data:
        logger.info("Stopping Mission")
        stop_mission()
    return ('', 204)


if __name__ == "__main__": 
    signal.signal(signal.SIGINT, handler_signals)
    signal.signal(signal.SIGTERM, handler_signals)
    app.run(host='0.0.0.0', port=8000)