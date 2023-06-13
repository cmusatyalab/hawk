# SPDX-FileCopyrightText: 2022,2023 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

# Usage: python home_flutter.py 

import base64
import io
import json
import multiprocessing as mp
import os
import signal
import socket
import sys
import threading
import time
import yaml
import zmq
from pathlib import Path
from pprint import pprint

from admin import Admin 
from datetime import datetime
from flask import Flask, request, make_response, jsonify
from hawk.ports import H2A_PORT, H2C_PORT
from home import *
from inbound import InboundProcess
from logzero import logger
from outbound import OutboundProcess
from pathlib import Path
from PIL import Image
from script_labeler import ScriptLabeler
from typing import Iterable
from ui_labeler import UILabeler
from utils import define_scope, write_config, get_ip

REMOTE_USER = 'root'
CONFIG = os.getenv('HOME')+"/.hawk/config.yml" # PUT this in .hawk
home_path = Path(__file__).resolve().parent
app_data = {}
app = Flask(__name__)


def restart_scouts(scouts: Iterable[str]):
    host_file = home_path / 'hosts'
    host_file.write_text("\n".join(scouts))
    restart_file = home_path / 'scout-restart.sh'

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
   #result_cmd = f"python {home_path}/result_stream.py {image_root}"
    result_cmd = f"python {home_path}/result_stream_new.py {image_root}"
    logger.info(result_cmd)
    os.system(result_cmd)
    return


def configure_mission(filter_config):
    # Stop previous missions
    stop_mission()

    config_path = sys.argv[1] if len(sys.argv) > 1 \
        else (Path.cwd() / 'configs/config.yml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
        #logger.info(config)
        #pprint(config)
        pprint(config['train_strategy'])

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

   ### Removing all bandwidth restrictions
    
    bandwidth = config.get('bandwidth', "100")
    assert int(bandwidth) in [100, 30, 12], "Fireqos script may not exist for {}".format(bandwidth)
    config['bandwidth'] = ["[[-1, \"{}k\"]]".format(bandwidth) for _ in config['scouts']] 

    config['train_strategy']['type'] = filter_config['name']
    logger.info(f"Train strategy is: {config['train_strategy']}")

    # Add more filters here 
    if filter_config['name'] == "fsl":
        support_string = filter_config['args']['support']
        image = Image.open(io.BytesIO(base64.b64decode(support_string)))
        # Resize to 256 x 256
        dim = (256, 256)
        image = image.resize(dim, Image.LANCZOS)
        image_path = "/home/eric/.hawk/current_mission.jpg" ## This is the support set image.
        image.save(image_path)
        config['train_strategy']['example_path'] = image_path
    elif filter_config['name'] == 'dnn_classifier':
        #init_model = config['train_strategy']['bootstrap_path']
        pass
        ### find the initial model
        ### find the initial dataset

    else:
        raise NotImplementedError("Unknown train_strategy {}".format(filter_config['name']))

    # create local directories 
    mission_dir = Path(config['home-params']['mission_dir'])
    mission_dir = mission_dir / mission_id
    logger.info(mission_dir)

    global log_dir
    log_dir = mission_dir / 'logs'
    config['home-params']['log_dir'] = str(log_dir)
    end_file = log_dir / 'end'

    global meta_dir, label_dir
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
    scouts = config['scouts']
    scout_ips = [socket.gethostbyname(scout) for scout in scouts]

    app_data['scouts'] = scouts
    restart_scouts(scouts)

    processes = [] 
    stop_event = mp.Event()
    
    meta_q = mp.Queue()
    global label_q
    label_q = mp.Queue()
    stats_q = mp.Queue()
    stats_q.put((0,0,0))

    thread = threading.Thread(target = get_results)
    thread.start()
    
    try:

        # Starting home to admin conn 
        context = zmq.Context()
        h2a_socket = context.socket(zmq.REQ)
        h2a_socket.connect(f"tcp://127.0.0.1:{H2A_PORT}")

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
        h2c_port = config.get('h2c_port', H2C_PORT)
        home_outbound = OutboundProcess()
        p = mp.Process(target=home_outbound.send_labels, kwargs={'scout_ips': scout_ips,
                                                                 'h2c_port': h2c_port,
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
        #stop_mission()
        logger.info("killing all chld processes...")
        logger.error(e)



@app.route('/hawk_push_labels', methods = ['POST'])
def label_Sample():
    label_nums = {"positive": "1", "negative": "0"}
    print("In label function...")
    request_data = request.data #getting the response data
    request_data = json.loads(request_data.decode('utf-8'))
    #label_sign = request_data['label_sign']
    #image_strings = request_data['label_strings']
    for key, val in request_data.items():
        print(key, val)
    #response = f'{label_sign}'
    for sample, label in request_data.items():
        print(sample)
        tile_index = sample.split("/")[-1].split(".")[0]
        meta_path = meta_dir / f"{tile_index}.json"
        print(meta_path)
        with open(meta_path, 'r') as f:
            meta_data = json.load(f)
        del meta_data['score']
        meta_data['imageLabel'] = label_nums[label]
        meta_data['boundingBoxes'] = '[]'
        label_path = label_dir / f"{tile_index}.json"
        with open(label_path, "w") as f:
            json.dump(meta_data, f)
        label_q.put(label_path)
        print("put in out queue")

        ## Still need to figure out a method to store positives and negatives labeled for stats

    ### now that i have the sign and labels, 
    ### Send label to outbound queue, save label in labels directory and any other admin tasks as in original ui labeler.

    app_data['started'] = True
    return ('', 204)

@app.errorhandler(500)
def internal_error(error):

    return "500 error"

@app.route('/get_stats', methods = ['GET'])
def get_stat():
    print("In stats_function...")
    #request_data = request.data #getting the response data
    #request_data = json.loads(request_data.decode('utf-8'))
    #label_sign = request_data['label_sign']
    #image_strings = request_data['label_strings']
    logs = os.listdir(log_dir)
    json_logs = [log for log in logs if log.endswith(".json")]
    index = len(json_logs)
    file_name = f"stats-{index:06d}.json"
    with open (os.path.join(log_dir,file_name), "r") as f:
        data = json.load(f)
    print(file_name)
    print(data)
    response = jsonify(data)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return  response
    
    


@app.route('/start', methods = ['POST'])
def startMission():
    print("In start mission...")
    name_dict = {'fsl':'fsl', 'hawk': 'dnn_classifier'}
    request_data = request.data #getting the response data
    request_data = json.loads(request_data.decode('utf-8'))
    print(request_data['name'])
    request_data['name'] = name_dict[request_data['name']]
    configure_mission(request_data)
    print("called configure mission...")
    app_data['started'] = True
    response = make_response('', 204)
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


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
    filter_config = {}
    signal.signal(signal.SIGINT, handler_signals)
    signal.signal(signal.SIGTERM, handler_signals)
    #filter_config['name'] = 'dnn_classifier'
    #configure_mission(filter_config)
    app.run(host='0.0.0.0', port=8000)
    print("Executed app.run...")

