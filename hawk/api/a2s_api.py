# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import copy
import gc
import glob
import io
import os
import time
from pathlib import Path
import torch
from logzero import logger
import json
import shlex
import subprocess
import zipfile

from google.protobuf import json_format

from hawk import api 
from hawk.core.hawk_stub import HawkStub
from hawk.core.mission import Mission
from hawk.core.utils import log_exceptions
from hawk.retrain.absolute_threshold_policy import AbsoluteThresholdPolicy                          
from hawk.retrain.percentage_threshold_policy import PercentageThresholdPolicy                      
from hawk.retrain.model_policy import ModelPolicy 
from hawk.retrain.retrain_policy import RetrainPolicy
from hawk.retrieval.tile_retriever import TileRetriever                                             
from hawk.retrieval.filesystem_retriever import FileSystemRetriever                                             
from hawk.retrieval.random_retriever import RandomRetriever                                             
from hawk.retrieval.retriever import Retriever   
from hawk.selection.selector import Selector                                                        
from hawk.selection.threshold_selector import ThresholdSelector                                     
from hawk.selection.topk_selector import TopKSelector
from hawk.selection.top_reexamination_strategy import TopReexaminationStrategy                      
from hawk.selection.full_reexamination_strategy import FullReexaminationStrategy                    
from hawk.selection.no_reexamination_strategy import NoReexaminationStrategy                        
from hawk.selection.reexamination_strategy import ReexaminationStrategy    
from hawk.trainer.dnn_classifier.trainer import DNNClassifierTrainer 
from hawk.trainer.yolo.trainer import YOLOTrainer 
from hawk.proto.messages_pb2 import Dataset, ScoutConfiguration, MissionId, ImportModel, \
    RetrainPolicyConfig,  SelectiveConfig, ReexaminationStrategyConfig, MissionResults, MissionStats  
    
MODEL_FORMATS = ['pt', 'pth']

class A2SAPI(object):

    def __init__(self, port: int):  
        self._port = port
        self._mission = None
    
    @log_exceptions 
    def a2s_configure_scout(self, msg):
        try:
            request =  ScoutConfiguration()
            request.ParseFromString(msg)

            reply = self._a2s_configure_scout(request)
        except Exception as e:
            logger.exception(e)
            reply = ("ERROR: {}".format(e)).encode()
            raise e
        finally:
            return reply 

    @log_exceptions
    def a2s_start_mission(self):
        try:
            reply = self._a2s_start_mission()  
            return reply 
        except Exception as e:
            logger.exception(e)
            raise e
    
    @log_exceptions
    def a2s_stop_mission(self):
        try:
            reply = self._a2s_stop_mission()  
            return reply 
        except Exception as e:
            logger.exception(e)
            raise e

    @log_exceptions
    def a2s_get_mission_stats(self):
        try:
            reply = self._a2s_get_mission_stats()
            return reply
        except Exception as e:
            logger.exception(e)
            raise e

    @log_exceptions
    def a2s_new_model(self, msg):
        try:
            request = ImportModel()
            request.ParseFromString(msg)
            reply = self._a2s_new_model(request)  
            return reply 
        except Exception as e:
            logger.exception(e)
            raise e

    @log_exceptions
    def a2s_get_test_results(self, msg):
        try:
            test_path = msg
            logger.info("Testing {}".format(test_path))
            assert os.path.exists(test_path)
            reply = self._a2s_get_test_results(test_path)  
            return reply 
        except Exception as e:
            logger.exception(e)
            raise e

    @log_exceptions 
    def a2s_get_post_mission_archive(self):
        try:
            reply = self._a2s_get_post_mission_archive()
        except Exception as e:
            logger.exception(e)
            reply = b""
            raise e
        finally:
            return reply 

    @log_exceptions
    def _a2s_configure_scout(self, request: ScoutConfiguration):
        try:
            self._root_dir = Path(request.missionDirectory) / 'data'
            self._model_dir = Path(request.missionDirectory) / 'pretrained'
            assert self._root_dir.is_dir(), "Root directory does not exist"
            model_dir = self._root_dir / request.missionId / 'model'


            mission_id = MissionId(value=request.missionId)
            retrain_policy = self._get_retrain_policy(request.retrainPolicy, model_dir)
            host_ip = request.scouts[request.scoutIndex]
            scouts = [HawkStub(scout, api.S2S_PORT, host_ip) for scout in request.scouts]
            mission = Mission(mission_id, request.scoutIndex, scouts, 
                            request.homeIP, retrain_policy, 
                            self._root_dir / mission_id.value,
                            self._port, self._get_retriever(request.dataset),
                            self._get_selector(request.selector, request.reexamination),
                            request.bootstrapZip, request.initialModel, request.validate)
            self._mission = mission

            model = request.trainStrategy
            trainer = None
            if model.HasField('dnn_classifier'):
                config = model.dnn_classifier
                trainer = DNNClassifierTrainer(mission, config.args)
            elif model.HasField('yolo'):
                config = model.yolo
                trainer = YOLOTrainer(mission, config.args)
            else:
                raise NotImplementedError('unknown model: {}'.format(
                    json_format.MessageToJson(model)))

            self.trainer = trainer
            mission.setup_trainer(trainer)
            logger.info('Create mission with id {}'.format(
                request.missionId))
            
            # Only supports one bandwidth
            logger.info(request.bandwidthFunc)
            self._setup_bandwidth(request.bandwidthFunc[request.scoutIndex])
            if mission.enable_logfile:
                mission.log_file.write("{:.3f} {} SEARCH CREATED\n".format(
                    time.time() - mission.start_time, mission.host_name))

            reply = b"SUCCESS"
        except Exception as e:
            reply = ("ERROR: {}".format(e)).encode()
        return reply
    
    def _setup_bandwidth(self, bandwidth_func : str) -> None:
        bandwidth_map = {
            '100k': '/root/fireqos/scenario-100k.conf', 
            '30k': '/root/fireqos/scenario-30k.conf',
            '12k': '/root/fireqos/scenario-12k.conf',
        }
        logger.info(bandwidth_func)
        bandwidth_list = json.loads(bandwidth_func)    
        default_file = bandwidth_map['100k']
        # bandwidth_file = default_file
        for time_stamp, bandwidth in bandwidth_list:
            bandwidth_file = bandwidth_map.get(bandwidth.lower(), default_file)

        # start fireqos
        bandwidth_cmd = "fireqos start {}".format(bandwidth_file)
        b = subprocess.Popen(shlex.split(bandwidth_cmd))
        b.communicate()
        return 

    def _a2s_start_mission(self):
        try:
            mission = self._mission
            mission_id = mission.mission_id.value
            logger.info('Starting mission with id {}'.format(mission_id))
            mission.start()
            if mission.enable_logfile:
                mission.log_file.write("{:.3f} {} SEARCH STARTED\n".format(
                    time.time() - mission.start_time, mission.host_name))

            reply = b"SUCCESS"
        except Exception as e:
            reply = ("ERROR: {}".format(e)).encode()
        return reply
    
    def _a2s_stop_mission(self):
        try:
            mission = self._mission

            if mission is None:
                return b"ERROR: Mission does not exist"

            mission_id = mission.mission_id.value
            logger.info('Stopping mission with id {}'.format(mission_id))
            if mission.enable_logfile:
                mission.log_file.write("{:.3f} {} SEARCH STOPPED\n".format(
                    time.time() - mission.start_time, mission.host_name))
            mission.stop()

            reply = b"SUCCESS"
        except Exception as e:
            reply = ("ERROR: {}".format(e)).encode()
        finally:
            # Stop fireqos
            bandwidth_cmd = "fireqos stop"
            b = subprocess.Popen(shlex.split(bandwidth_cmd))
            b.communicate()
            torch.cuda.empty_cache()
            gc.collect()
            self._mission = None

        return reply

    def _a2s_get_mission_stats(self):
        try:
            mission = self._mission
            if not mission:
                return api.Empty

            time_now = time.time() - mission.start_time

            if mission.enable_logfile:
                mission.log_file.write("{:.3f} {} SEARCH STATS\n".format(
                    time.time() - mission.start_time, mission.host_name))

            retriever_stats = mission.retriever.get_stats()
            selector_stats = mission.selector.get_stats()
            processed_objects = retriever_stats.dropped_objects + selector_stats.processed_objects

            mission_stats = vars(copy.deepcopy(retriever_stats))
            mission_stats.update(vars(copy.deepcopy(selector_stats)))
            keys_to_remove = ['total_objects', 'processed_objects', 'dropped_objects',
                              'passed_objects', 'false_negatives']
            for k in list(mission_stats):
                v = mission_stats[k]
                mission_stats[k] = str(v)
                if k in keys_to_remove:
                    del mission_stats[k]

            mission_stats.update({
                'server_time': str(time_now),
                'version': str(mission._model.version),
                'msg': "stats",
                'ctime': str(time.ctime()),
                'server_positives': str(mission.positives),
                'server_negatives': str(mission.negatives),
                })

            mission.stats_file.write("{}\n".format(json.dumps(mission_stats)))
            mission.stats_file.flush()

            reply = MissionStats(totalObjects=int(retriever_stats.total_objects),
                              processedObjects=processed_objects,
                              droppedObjects=retriever_stats.dropped_objects + selector_stats.dropped_objects,
                              falseNegatives=retriever_stats.false_negatives + selector_stats.false_negatives,
                              others=mission_stats)

            if mission.enable_logfile:
                mission.log_file.write("{:.3f} {} SEARCH STATS\n".format(
                    time.time() - mission.start_time, mission.host_name))
            
            reply = reply.SerializeToString()            
        except Exception as e:
            reply = ("ERROR: {}".format(e)).encode()
        return reply

            
    def _a2s_new_model(self, request: ImportModel):
        try:
            mission = self._mission
            model = request.model
            path = request.path
            version = model.version
            mission.import_model(model.content, path, version)
            logger.info("[IMPORT] FINISHED Model Import")
            if mission.enable_logfile:
                mission.log_file.write("{:.3f} {} IMPORT MODEL\n".format(
                    time.time() - mission.start_time, mission.host_name))

            reply = b"SUCCESS"
        except Exception as e:
            reply = ("ERROR: {}".format(e)).encode()
        return reply

    def _a2s_get_test_results(self, request: str):
        try:
            test_path = Path(request)
            
            if not test_path.is_file():
                raise Exception 
            
            mission = self._mission
            model_dir = str(mission.model_dir)
            files = sorted(glob.glob(os.path.join(model_dir, '*.*'))) 
            model_paths = [x for x in files if x.split('.')[-1].lower() in MODEL_FORMATS]
            logger.info(model_paths)
            
            def get_version(path, idx):
                name = path.name
                try:
                    version = int(name.split('model-')[-1].split('.')[0])
                except:
                    version = idx
                
                return version     
            
            results = {}
            for idx, path in enumerate(model_paths):
                path = Path(path)
                version = get_version(path, idx)
                logger.info("model {} version {}".format(path, version))
                # create trainer and check
                # model = mission.load_model(path, version=version)
                model = self.trainer.load_model(path, version=version)
                result = model.evaluate_model(test_path)
                results[version] = result
                
            reply = MissionResults(results=results)
            reply = reply.SerializeToString()            
        except Exception as e:
            reply = ("ERROR: {}".format(e)).encode()
        return reply


    def _a2s_get_post_mission_archive(self):
        try:
            mission = self._mission
            data_dir = mission.data_dir
            model_dir = mission.model_dir
           
            mission_archive = io.BytesIO() 
            with zipfile.ZipFile(mission_archive, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                for dirname, subdirs, files in os.walk(model_dir):
                    zf.write(dirname)
                    for filename in files:
                        zf.write(os.path.join(dirname, filename))

            logger.info("[IMPORT] FINISHED Archiving Mission models")

            mission_archive.seek(0)
            reply = mission_archive
        except Exception as e:
            reply = b""
        return reply
    
    def _get_retrain_policy(self, retrain_policy: RetrainPolicyConfig, model_dir: Path) -> RetrainPolicy:
        if retrain_policy.HasField('absolute'):
            return AbsoluteThresholdPolicy(retrain_policy.absolute.threshold,
                                           retrain_policy.absolute.onlyPositives)
        elif retrain_policy.HasField('percentage'):
            return PercentageThresholdPolicy(retrain_policy.percentage.threshold,
                                             retrain_policy.percentage.onlyPositives)
        elif retrain_policy.HasField('model'):
            logger.info("Model Policy")
            return ModelPolicy(str(model_dir))

        else:
            raise NotImplementedError('unknown retrain policy: {}'.format(
                json_format.MessageToJson(retrain_policy)))

    def _get_selector(self, selector: SelectiveConfig, reexamination_strategy: ReexaminationStrategyConfig) -> Selector:
        
        if selector.HasField('topk'):
            top_k_param = json_format.MessageToDict(selector.topk)
            logger.info("TopK Params {}".format(top_k_param))
            return TopKSelector(selector.topk.k, selector.topk.batchSize,
                                self._get_reexamination_strategy(
                                    reexamination_strategy))
        elif selector.HasField('threshold'):
            return ThresholdSelector(selector.threshold.threshold,
                                self._get_reexamination_strategy(
                                    reexamination_strategy))
        else:
            raise NotImplementedError('unknown selector: {}'.format(
                json_format.MessageToJson(selector)))

    def _get_reexamination_strategy(self, reexamination_strategy: ReexaminationStrategyConfig) -> ReexaminationStrategy:
        reexamination_type = reexamination_strategy.type
        if reexamination_type == 'none':
            return NoReexaminationStrategy()
        elif reexamination_type == 'top':
            return TopReexaminationStrategy(reexamination_strategy.k)
        elif reexamination_type == 'full':
            return FullReexaminationStrategy()
        else:
            raise NotImplementedError(
                'unknown reexamination strategy: {}'.format(
                    json_format.MessageToJson(reexamination_strategy)))

    def _get_retriever(self, dataset: Dataset) -> Retriever:
        if dataset.HasField('tile'):
            return TileRetriever(dataset.tile)
        if dataset.HasField('filesystem'):
            return FileSystemRetriever(dataset.filesystem)
        elif dataset.HasField('random'):
            return RandomRetriever(dataset.random)
        else:
            raise NotImplementedError('unknown dataset: {}'.format(
                json_format.MessageToJson(dataset)))
