# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import base64
import os
import queue
import io
import json
import socket
import sys
import threading
import time
import yaml                                                                                              
import zmq
  
from collections import defaultdict                                                                      
from logzero import logger
from google.protobuf.json_format import MessageToDict                                                             
from pathlib import Path
from PIL import Image

from home import ZFILL
from hawk.api import H2A_PORT, A2S_PORT 
from hawk.proto.messages_pb2 import *


import yaml 

LOG_INTERVAL = 60

class Admin:

    def __init__(self, home_ip, mission_id, explicit_start=False) -> None:
        self._home_ip = home_ip
        self._start_event = threading.Event()
        self.last_stats = (0,0,0)
        self._mission_id = mission_id
        self.explicit_start = explicit_start
        self.scout_stubs = {}
        self.test_path = ""

    def receive_from_home(self, stop_event, stats_q):

        self.stats_q = stats_q
        self.stop_event = stop_event
        
        # Bind H2A Server
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f'tcp://127.0.0.1:{H2A_PORT}')

        while not stop_event.is_set():
            msg_string = socket.recv_string()
            header, body = msg_string.split()
            socket.send_string("RECEIVED")
        
            if header == "config":
                config_path = body 
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                self._setup_mission(config)

            else:
                raise NotImplementedError("Unknown header {}".format(header))
        
    
    def _setup_mission(self, config):

        self._mission_name = config['mission-name']    
        self.log_dir = Path(config['home-params']['log_dir'])
        self.end_file = self.log_dir / 'end'
        self.end_time = int(config.get('end-time', 5000))
        scouts = config['scouts']
        self.scouts = scouts
        
        # homeIP 
        home_ip = self._home_ip
        
        # trainLocation 
        train_location = config['train-location']

        # missionDirectory
        mission_directory = config['scout-params']['mission_dir']
        self.test_path = config['scout-params'].get('test_path', "")
        
        # trainStrategy
        train_config = config['train_strategy']
        train_type = train_config['type']
       
        train_strategy = None 
        if train_type == "dnn_classifier":
            default_args = {'mode': "hawk",
                    "unfreeze_layers": "3",
                    "arch": "resnet50",
                    "online_epochs": "[[10,0],[15,100]]"}
            train_strategy = TrainConfig(
                dnn_classifier=ModelConfig(
                    args=train_config.get('args', default_args),
                )
            )
        elif train_type == "yolo":
            default_args = {'mode': "hawk",
                    "online_epochs": "[[10,0],[15,100]]"}
            train_strategy = TrainConfig(
                yolo=ModelConfig(
                    args=train_config.get('args', default_args),
                )
            )
        elif train_type == "fsl":
            support_path = train_config['example_path']
            image = Image.open(support_path).convert('RGB')
            content = io.BytesIO()
            image.save(content, format='JPEG', quality=75)
            content = content.getvalue()
            support_data = base64.b64encode(content).decode('utf8')
            default_args = {'mode': "hawk",
                    "support_path": "/srv/diamond/dota/support.jpg", 
                    "fsl_traindir": "/srv/diamond/dota/fsl_traindir",
                    "support_data": support_data}

            train_strategy = TrainConfig(
                fsl=ModelConfig(
                    args=default_args,
                    # args=train_config.get('args', default_args),
                )
            )
        else:
            raise NotImplementedError("Unknown train strategy {}".format(train_type))
        
        # retrainPolicy
        retrain_config = config['retrain_policy']
        retrain_type = retrain_config['type']
        if  retrain_type == "percentage":
            retrain_policy = RetrainPolicyConfig(
                                percentage=PercentagePolicyConfig(
                                    threshold=retrain_config['threshold'],
                                    onlyPositives=retrain_config['only_positives'],))
        elif retrain_type == "absolute":
            retrain_policy = RetrainPolicyConfig(
                                absolute=AbsolutePolicyConfig(
                                    threshold=retrain_config['threshold'],
                                    onlyPositives=retrain_config['only_positives'],))
        else:
            raise NotImplementedError("Unknown retrain policy {}".format(retrain_type))
          
        # dataset
        dataset_config = config['dataset']
        dataset_type = dataset_config['type']
        if dataset_type == "video":
            video_file_list = dataset_config["video_list"]
        logger.info("Index {}".format(dataset_config['index_path']))
        timeout = dataset_config.get('timeout', 20)

        dataset = []
        for i, scout in enumerate(scouts):
            if dataset_type == "tile":
                dataset.append(Dataset(
                    tile=FileDataset(
                        dataPath=dataset_config['index_path'],
                        timeout=timeout,
                    )
                ))
            elif dataset_type == "filesystem":
                dataset.append(Dataset(
                    filesystem=FileDataset(
                        dataPath=dataset_config['index_path'],
                        tileSize=dataset_config['tile_size'],
                        timeout=timeout,
                    )
                ))
            elif dataset_type == "frame":
                dataset.append(Dataset(
                    frame=FileDataset(
                        dataPath=dataset_config['index_path'],
                        tileSize=dataset_config['tile_size'],
                        timeout=timeout,
                    )
                )) 
            elif dataset_type == "random":
                dataset.append(Dataset(
                    random=FileDataset(
                        dataPath=dataset_config['index_path'],
                        numTiles= int(dataset_config.get('tiles_per_frame', 200))
                        timeout=timeout,
                    )
                ))
            elif dataset_type == "cookie":                                                               
                dataset.append(Dataset(
                    random=FileDataset(
                        dataPath=dataset_config['index_path'],
                        numTiles=int(dataset_config.get('tiles_per_frame', 200)),
                        timeout=timeout,
                    )
                ))
            elif dataset_type == "video":
                dataset.append(Dataset(
                    video=Streaming_Video(
                        video_path=video_file_list[i],
                        sampling_rate_fps=1,
                        width=4000,
                        height=3000,
                        tile_width=250,
                        tile_height=250,
                        timeout=timeout,
                    )
                ))
            else:
                raise NotImplementedError("Unknown dataset {}".format(dataset_type))

        # reexamination
        reexamination_config = config['reexamination']
        reexamination_type = reexamination_config.get('type', 'top')
        
        reexamination = None 
        if reexamination_type == "top":
            k_value = reexamination_config.get('k', 100)
            reexamination = ReexaminationStrategyConfig(
                type=reexamination_type,
                k=k_value,
                )
        elif reexamination_type == "full" or reexamination_type == "none":
            reexamination = ReexaminationStrategyConfig(
                type=reexamination_type,
                )
        else:
            raise NotImplementedError("Unknown reexamination {}".format(reexamination_type))
        
        
        # selector
        selector_config = config['selector']
        selector_type = selector_config.get('type', 'topk')
        
        if selector_type == "topk":
            topk_config = selector_config.get('topk', {})
            k_value = topk_config.get('k', 10)
            batch_size = topk_config.get('batchSize', 1000)
            selector = SelectiveConfig(
                topk=TopKConfig(
                    k=k_value,
                    batchSize=batch_size, 
                )
            )
        elif selector_type == "token":
            token_config = selector_config.get('token', {})
            init_samples = token_config.get('initial_samples')
            batch_size = token_config.get('batch_size')
            selector = SelectiveConfig(
                token=TokenConfig(
                    initial_samples=init_samples,
                    batch_size=batch_size
                )
            )
        else:
            raise NotImplementedError("Unknown selector {}".format(selector_type))
        
        # initialModel
        model_path = train_config.get('initial_model_path','')
        model_content = b''
        if os.path.isfile(model_path):
            with open(model_path, 'rb') as f:
                model_content = f.read()
       
        initial_model = None 
        if len(model_content):    
            initial_model = ModelArchive(
                content=model_content,      
            )
       
        # bootstrapZip
        bootstrap_path = train_config.get('bootstrap_path', '')
        bootstrap_zip = b''
        if os.path.exists(bootstrap_path):
            with open(bootstrap_path, 'rb') as f:
                bootstrap_zip = f.read()
       
        # bandwidthFunc
        bandwidth_config = config['bandwidth']
        assert len(scouts) == len(bandwidth_config), "Length Bandwidth {} does not match {}".format(
            len(bandwidth_config), len(scouts)
        )
        bandwidth_func = {}
        for i, _b in enumerate(bandwidth_config):
            bandwidth_func[int(i)] = str(_b)  

        train_validate = train_config.get('validate', True)
         
        self.scout_stubs = {}
        for i, scout in enumerate(scouts):
            ip = socket.gethostbyname(scout)
            context = zmq.Context()
            stub = context.socket(zmq.REQ)
            stub.connect("tcp://{}:{}".format(ip, A2S_PORT))
            self.scout_stubs[i] = stub
       
        # setup ScoutConfiguration
        # Call a2s_configure_scout and wait for success message 
        return_msgs = []
        logger.info(self._mission_id)
        logger.info(scouts)
        for index, stub in self.scout_stubs.items():
            scout_config = ScoutConfiguration(
                missionId=self._mission_id, 
                scouts=scouts, 
                scoutIndex=index, 
                homeIP=home_ip, 
                trainLocation=train_location, 
                missionDirectory=mission_directory,
                trainStrategy=train_strategy,
                retrainPolicy=retrain_policy,
                dataset=dataset[index],
                selector=selector, 
                reexamination=reexamination,
                initialModel=initial_model, 
                bootstrapZip=bootstrap_zip,
                bandwidthFunc=bandwidth_func,
                validate=train_validate,
            )
            msg = {
                "method": "a2s_configure_scout",
                "msg": scout_config.SerializeToString()
            }
            stub.send_pyobj(msg)
        
        for index, stub in self.scout_stubs.items():
            reply = stub.recv()
            return_msgs.append(reply.decode())
       
        # Remove scouts that failed
         
        for i, msg in enumerate(return_msgs):
            if "ERROR" in msg:
                logger.error(f"ERROR during Configuration from Scout {i} \n {msg}")
                del self.scout_stubs[i]
        
        if not self.explicit_start: 
            self.start_mission()        

        return "SUCCESS"
        
        
    def start_mission(self):
        """Explicit start Mission command""" 
        # Start Mission  

        logger.info("Starting mission")
        self.log_files = [open(os.path.join(str(self.log_dir), 'get-stats-{}.txt'.format(i)), "a") 
                          for i, stub in enumerate(self.scout_stubs)]
        self.start_time = time.time()
        for index, stub in self.scout_stubs.items():
            msg = {
                "method": "a2s_start_mission",
                "msg": b"",
            }
            stub.send_pyobj(msg)

        for index, stub in self.scout_stubs.items():
            stub.recv()

        logger.info("Start msg received")
        
        threading.Thread(target=self.get_mission_stats, name='get-stats').start()
        return         

    def stop_mission(self):
        """Explicit stop Mission command""" 

        for index, stub in self.scout_stubs.items():
            msg = {
                "method": "a2s_stop_mission",
                "msg": b"",
            }
            stub.send_pyobj(msg)

        for index, stub in self.scout_stubs.items():
            stub.recv()
        return         

    def get_mission_stats(self):
        time.sleep(10)
        count = 1
        break_count = 0
        prev_bytes = 0
        prev_processed = 0
        try:
            while not self.stop_event.is_set():
                stats = self.accumulate_mission_stats()
                last_stats = None
                while True:
                    try:
                        last_stats = self.stats_q.get_nowait()
                    except queue.Empty:  
                        break
                
                if last_stats is None:
                    last_stats = self.last_stats
                else:
                    self.last_stats = last_stats
                keys = ['positives', 'negatives', 'bytes']
                for key, value in zip(keys, last_stats):
                    stats[key] = value

                log_path = self.log_dir / "stats-{}.json".format(str(count).zfill(ZFILL))
                with open(log_path, "w") as f:
                    stats['home_time'] = time.time() - self.start_time
                    json.dump(stats, f, indent=4, sort_keys=True)

                if stats['home_time'] > self.end_time:
                    logger.info("End mission")
                    with open(self.end_file, "w") as f:
                        f.write("\n")
                    break

                if stats['processedObjects'] != 0: 
                    if (stats['processedObjects'] == stats['totalObjects'] or
                        (stats['retrieved_tiles'] == stats['totalObjects'] and
                        prev_bytes == stats['bytes'] and prev_processed == stats['processedObjects'])):
                        self.stop_event.set()
                        logger.info("End mission")
                        with open(self.end_file, "w") as f:
                            f.write("\n")
                        break

                prev_bytes = stats['bytes']
                prev_processed = stats['processedObjects']
                count += 1            
                time.sleep(LOG_INTERVAL)
        except (Exception, KeyboardInterrupt) as e:
            raise e
        self.stop_event.set()
        self.stop_mission()
        return 
    
    def get_post_mission_archive(self):
        for index, stub in self.scout_stubs.items():
            msg = {
                "method": "a2s_get_post_mission_archive",
                "msg": b"",
            }
            stub.send_pyobj(msg)
            reply = stub.recv()

            if len(reply):
                with open("mission_{}.zip".format(index), "wb") as f:
                    f.write(reply)
        return 
    
            
    def get_test_results(self):
        assert len(self.test_path), "Test path not provided"
        for index, stub in self.scout_stubs.items():
            msg = {
                "method": "a2s_get_test_results",
                "msg": self.test_path,
            }
            stub.send_pyobj(msg)

        for index, stub in self.scout_stubs.items():
            reply = stub.recv()
            hostname = self.scouts[index].split('.')[0]
            results_dir = (Path(self.log_dir.parent)/ 
                           "results" / 
                           "{}".format(hostname)) 
            results_dir.mkdir(parents=True, exist_ok=True)            
            if len(reply):
                try:
                    mission_stat = MissionResults()
                    mission_stat.ParseFromString(reply)
                    mission_stat = mission_stat.results
                
                    for version, result in mission_stat.items():
                        model_stat = MessageToDict(result)
                        stat_path = (results_dir / 
                                     "model-result-{}.json".format(str(version).zfill(ZFILL)))
                        with open(stat_path, "w") as f:
                            json.dump(model_stat, f, indent=4, sort_keys=True)
                except Exception as e:
                    msg = reply.decode()
                    logger.error(f"ERROR during Testing from Scout {index} \n {msg}")
        return 

    def accumulate_mission_stats(self):
        stats = defaultdict(lambda: 0)
        str_ignore = ['server_time', 'ctime',
                      'train_positives', 'server_positives',
                      'msg']
        single = ['server_time', 'train_positives', 'version']
        for index, stub in self.scout_stubs.items():
            msg = {
                "method": "a2s_get_mission_stats",
                "msg": b"",
            }
            stub.send_pyobj(msg)

        for index, stub in self.scout_stubs.items():
            reply = stub.recv()
            mission_stat = MissionStats()
            mission_stat.ParseFromString(reply)
            mission_stat = MessageToDict(mission_stat)
            self.log_files[index].write("{}\n".format(json.dumps(mission_stat)))
            for k, v in mission_stat.items():
                if isinstance(v, dict):
                    others = v
                    for key, value in others.items():
                        if key in mission_stat:
                            continue
                        if key in str_ignore:
                            if index == 0:
                                stats[key] = value
                        elif key in single:
                            if index == 0:
                                stats[key] = float(value)
                        else:
                            stats[key] += float(value)
                else:
                    stats[k] += float(v)
        return stats
