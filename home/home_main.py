# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import os
import shutil
import signal
import sys
import time
import yaml
import zmq
import multiprocessing as mp
from datetime import datetime

from admin import Admin 
from home import *
from script_labeler import ScriptLabeler
from inbound import InboundProcess
from outbound import OutboundProcess
from logzero import logger
from pathlib import Path
from hawk.core.utils import get_ip
from hawk.api import  H2A_PORT

# Usage: python home_main.py config/config.yml
def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 \
                    else (Path.cwd() / 'configs/config.yml')

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Setting up mission 
    mission_name = config.get('mission-name', "test")
    
    assert (Path(config['train_strategy'].get('initial_model_path', '')).exists() or 
            Path(config['train_strategy'].get('bootstrap_path', '')).exists())    

    mission_id = "_".join([mission_name, 
                               datetime.now().strftime('%Y%m%d-%H%M%S')])
   
    bandwidth = config.get('bandwidth', "100")
    assert int(bandwidth) in [100, 30, 12], "Fireqos script may not exist for {}".format(bandwidth)
    config['bandwidth'] = ["[[-1, \"{}k\"]]".format(bandwidth) for _ in config['scouts']] 

    # create local directories 
    mission_dir = Path(config['home-params']['mission_dir'])
    mission_dir = mission_dir / mission_id
    print(mission_dir)

    log_dir = mission_dir / 'logs'
    config['home-params']['log_dir'] = str(log_dir)
    end_file = log_dir / 'end'

    image_dir = mission_dir / 'images'
    meta_dir = mission_dir / 'meta'
    label_dir = mission_dir / 'labels'

    log_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    # Save config file to log_dir
    config_path = log_dir / 'hawk.yml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # Setting up helpers
    scout_ips = config['scouts']
    processes = [] 
    stop_event = mp.Event()
    meta_q = mp.Queue()
    label_q = mp.Queue()
    stats_q = mp.Queue()
    stats_q.put((0,0,0))
    
    try:

        # Starting home to admin conn 
        context = zmq.Context()
        h2a_socket = context.socket(zmq.REQ)
        h2a_socket.connect(f'tcp://127.0.0.1:{H2A_PORT}')

        # Setup admin 
        home_ip = get_ip()

        logger.info("Starting Admin Process")
        home_admin = Admin(home_ip, mission_id)
        p = mp.Process(target=home_admin.receive_from_home, kwargs={'stop_event': stop_event,
                                                                    'stats_q': stats_q})
        processes.append(p)
        p.start()
    
        # Start inbound process
        logger.info("Starting Inbound Process")
        train_location = config['train-location']
        home_inbound = InboundProcess(image_dir, 
                                      meta_dir, 
                                      train_location)
        p = mp.Process(target=home_inbound.receive_data, kwargs={'result_q': meta_q,
                                                                 'stop_event': stop_event})
        processes.append(p)
        p.start()
        
        # Start labeler process 
        logger.info("Starting Labeler Process")
        labeler =  config.get('labeler', 'script')
        gt_dir = config['home-params'].get('label_dir', "")
        trainer = (config['train_strategy']['type']).lower()
        logger.info("Trainer {}".format(trainer))
        label_mode = "classify"
        
        if trainer == "yolo":
            label_mode = "detect"
            
        if labeler == 'script':
            home_labeler = ScriptLabeler(label_dir, 
                                         gt_dir,
                                         label_mode)
        else:
            raise NotImplementedError("Labeler {} not implemented".format(labeler))

        p = mp.Process(target=home_labeler.start_labeling, kwargs={'input_q': meta_q,
                                                                   'result_q': label_q,
                                                                   'stats_q': stats_q,
                                                                   'stop_event': stop_event})
        p.start()
        processes.append(p)
    
        # start outbound process  
        logger.info("Starting Outbound Process")
        home_outbound = OutboundProcess()
        p = mp.Process(target=home_outbound.send_labels, kwargs={'scout_ips': scout_ips,
                                                                 'result_q': label_q,
                                                                 'stop_event': stop_event})
        p.start()
        processes.append(p)

        # Send config file to admin 
        # send msg "<config> <path to config file>"
        
        h2a_socket.send_string(f"config {config_path}")
        h2a_socket.recv()
   
        while not stop_event.is_set():
            if end_file.is_file():
                stop_event.set()
            time.sleep(10)

        logger.info("Stop event is set")
    except KeyboardInterrupt as e:
        logger.error(e)
    finally:
        stop_event.set()
        home_admin.stop_mission()
        time.sleep(10)
        for p in processes:
            p.terminate()
        pid = os.getpid()
        os.kill(pid, signal.SIGKILL)

        # Uncomment to delete mission dir after mission

        # if mission_dir.is_dir():
        #     logger.info("Deleting directory {}".format(mission_dir))
        #     shutil.rmtree(mission_dir)

if __name__ == "__main__": 
    main()
        
