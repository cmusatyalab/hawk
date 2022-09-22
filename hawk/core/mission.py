# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import io
import copy
import multiprocessing as mp
import os
import sys
import threading
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Tuple
import json

import numpy as np
import tensorboard
from google.protobuf.any_pb2 import Any
from logzero import logger
from torch.utils.tensorboard import SummaryWriter
from hawk.api.s2s_api import S2SServicer

from hawk.context.data_manager_context import DataManagerContext
from hawk.context.model_trainer_context import ModelContext
from hawk.core.data_manager import DataManager
from hawk.core.hawk_stub import HawkStub
from hawk.core.model import Model
from hawk.core.object_provider import ObjectProvider
from hawk.retrain.retrain_policy_base import RetrainPolicyBase
from hawk.retrieval.retriever import Retriever
from hawk.selection.selector_base import Selector
from hawk.selection.token_selector import TokenSelector
from hawk.api.s2s_api import S2SServicer, s2s_receive_request
from hawk.api.s2h_api import S2HPublisher 
from hawk.api.h2c_api import H2CSubscriber
from hawk.core.utils import log_exceptions, get_server_ids, get_ip
from hawk.proto.messages_pb2 import *


class Mission(DataManagerContext, ModelContext):

    def __init__(self, mission_id: MissionId, scout_index: int, scouts: List[HawkStub],
                 home_ip: str, retrain_policy: RetrainPolicyBase,
                 root_dir: Path, port: int, retriever: Retriever, selector: Selector, 
                 bootstrap_zip: bytes, initial_model: ModelArchive, 
                 validate: bool = False):

        super().__init__()
        logger.info("Initialization")
        self.start_time = time.time()
        self._id = mission_id
        self.positives = 0
        self.negatives = 0
        self._scout_index = scout_index
        self._scouts = scouts

        self._retrain_policy = retrain_policy
        self._data_dir = root_dir / 'data'
        self._log_dir = root_dir / 'tb'
        self._model_dir = root_dir / 'model'
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(self._model_dir, exist_ok=True)
        self.host_name = (get_server_ids()[0]).split('.')[0]
        self.host_ip = get_ip()
        self.home_ip = home_ip
        self.log_file = open(self._log_dir / 'log-{}.txt'.format(self.host_name), "a")
        self.result_path = str(self._log_dir / 'sent-{}.txt'.format(self.host_name))
        self.enable_logfile = True
        self.trainer = None
        self.trainer_type = None

        self._port = port

        self.retriever = retriever
        self.selector = selector
        self.bootstrap_zip = bootstrap_zip
        
        self.initial_model = initial_model
        self._validate = validate

        # Indicates that the mission will seed the strategy with an initial set
        # of examples. The strategy should therefore hold off on returning
        # inference results until its underlying model is trained
        self._has_initial_examples = True 

        self._model: Optional[Model] = None

        self._model_stats = TestResults(version=-1)
        self._model_lock = threading.Lock()
        self._initial_model_event = threading.Event()
        self._model_event = threading.Event()
        self._last_trained_version = -1
        self._object_model_excess = 2000

        self._test_results: Dict[int, List[List[Tuple[int, float]]]] = defaultdict(list)
        self._results_condition = mp.Condition()

        self._staged_models: Dict[int, Model] = {}
        self._staged_model_condition = mp.Condition()

        self._abort_event = threading.Event()

        self._data_manager = DataManager(self)
        self.selector.add_context(self)
        self.retriever.add_context(self)
        self.object_count = 0

        self.stats_count = 0 
        
        # setup_result_connections()
        logger.info("SETTING UP S2H API")
        s2h_conn, s2h_input = mp.Pipe(False)
        p = mp.Process(target=S2HPublisher.s2h_send_tiles, 
                       args=(self.home_ip, s2h_conn))
        p.start()

        self._result_thread = threading.Thread(target=self._get_results, 
                                               args=(s2h_input,), name='get-results')
        
        self._label_thread = None
        logger.info("SETTING UP H2C API")
        h2c_output, h2c_input = mp.Pipe(False)
        p = mp.Process(target=H2CSubscriber.h2c_receive_labels, args=(h2c_input,))
        self._label_thread = threading.Thread(target=self._get_labels, args=(h2c_output,), name='get-labels')
        p.start()
        logger.info("SETTING UP S2S Server {}".format(self._scout_index))
        s2s_input, s2s_output = mp.Queue(), mp.Queue()
        p = mp.Process(target=s2s_receive_request, args=(s2s_input, s2s_output,))
        self._s2s_thread = threading.Thread(target=self._s2s_process, 
                                            args=(s2s_input, s2s_output,), name='internal')
        p.start()
        
        self.start_time = time.time()
        s2s_object = S2SServicer(self)
        self.s2s_methods = dict((k, getattr(s2s_object, k))
                    for k in dir(s2s_object)
                    if callable(getattr(s2s_object, k)) and
                    not k.startswith('_'))


    def setup_trainer(self, trainer):
        logger.info("Setting up trainer")
        self.trainer = trainer
        threading.Thread(target=self._train_thread, name='train-model').start()
        self._retrain_policy.reset()
        self._model_event.set()

        # Wait for intial model to be trained
        while not self._model:
            pass
            

    def check_initial_model(self):
        if self.initial_model is None:
            return False
        return len(self.initial_model.content)

    def store_labeled_tile(self, tile: LabeledTile) -> None:
        self._data_manager.store_labeled_tile(tile)
        return

    def distribute_label(self, label: LabelWrapper) -> None:
        self._data_manager.distribute_label(label)
        return
        
    def get_example(self, example_set: DatasetSplit, label: str, example: str) -> Path:
        return self._data_manager.get_example_path(example_set, label, example)

    def get_test_results(self) -> TestResults:
        # We assume that the master is the source of truth
        with self._model_lock:
            stats = self._model_stats

        if stats is not None:
            return stats

        return TestResults(version=self._model.version)

    def load_model(self, model_path: str = "", content: bytes = b"", model_version: int = -1):
        logger.info("Loading model")
        model_path = Path(model_path)
        assert model_path.exists() or len(content)
        model = self.trainer.load_model(model_path, content, model_version)
        return model

    def import_model(self, model_path: str, content: bytes, model_version: int) -> None:
        model = self.load_model(model_path, content, model_version)
        self._set_model(model, False)
        return 

    def export_model(self) -> ModelArchive:
        assert self._scout_index == 0

        memory_file = io.BytesIO()
        with self._model_lock:
            assert self._model is not None
            model = self._model
            model_dump = model.serialize()
            model_version = model.version
            train_examples = model.train_examples

        with zipfile.ZipFile(memory_file, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('model', model_dump)

        memory_file.seek(0)
        return ModelArchive(content=memory_file.getvalue(),
                            version=model_version)

    def train_model(self, trainer_index: int=0) -> None:
        assert self._scout_index != 0 
        threading.Thread(target=self._train_model_slave_thread, name='train-model').start()

    def reset(self, train_only: bool) -> None:
        self._data_manager.reset(train_only)

        with self._model_lock:
            self._model = None
            self._model_stats = TestResults(version=-1)
            self._last_trained_version = -1

        self.selector.new_model(None)

    def get_last_trained_version(self) -> int:
        with self._model_lock:
            return self._last_trained_version

    @property
    def scout_index(self) -> int:
        return self._scout_index

    @property
    def scouts(self) -> List[HawkStub]:
        return self._scouts

    @property
    def port(self) -> int:
        return self._port

    @property
    def mission_id(self) -> MissionId:
        return self._id

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    def check_create_test(self):
        return self._validate

    def new_labels_callback(self, new_positives: int, new_negatives: int, retrain=True) -> None:
        if self._abort_event.is_set():
            return 
        
        end_t = time.time()

        if retrain:
            self.positives += new_positives
            self.negatives += new_negatives

        self._retrain_policy.update(new_positives, new_negatives)
        if self.enable_logfile:
            self.log_file.write("{:.3f} {} {}_NEW Examples \
                Positives {} Negatives {}\n".format(end_t - self.start_time, time.ctime(), 
                                                    self.host_name,
                                                    self._retrain_policy.positives,
                                                    self._retrain_policy.negatives))

            self.log_file.flush()

        if not retrain:
            return

        if self._retrain_policy.should_retrain():
            should_retrain = True
        else:
            with self._model_lock:
                model = self._model
            should_retrain = model is None

        if should_retrain:
            self._retrain_policy.reset()
            self._model_event.set()

    def start(self) -> None:
        try:
            self.retriever.start()
            threading.Thread(target=self._retriever_thread, name='get-objects').start()
            threading.Thread(target=self._infer_thread, name='infer-results').start()
            self._result_thread.start()
            self._label_thread.start()
            self._s2s_thread.start()
            
            self.start_time = time.time()
        except Exception as e:
            self.retriever.stop()
            self.selector.clear()
            self.stop()

    def stop(self) -> None:
        if not self._abort_event.is_set():
            if self._model is not None and self.enable_logfile:
                self.log_file.write("{:.3f} {}_{} SEARCH STOPPED\n".format(
                    time.time() - self.start_time,
                    self.host_name, self._model.version))
        # Stop Mission
        self.enable_logfile = False
        self._abort_event.set()
        self.retriever.stop()
        self.selector.clear()
        self.log_file.close()
        sys.exit(0)

    def get_example_directory(self, example_set: DatasetSplit): 
        return self._data_manager.get_example_directory(example_set) 

    def _objects_for_model_version(self) -> Iterable[Optional[ObjectProvider]]:
        if self._abort_event.is_set(): 
            return

        with self._model_lock:
            starting_version = self._model.version if self._model is not None else None
            model = self._model

        logger.info('Starting evaluation with model version {}'.format(starting_version))

        while not self._abort_event.is_set():
            with self._model_lock:
                version = self._model.version if self._model is not None else None
            if version != starting_version:
                logger.info('Done evaluating with model version {}  \
                    (new version {} available)'.format(starting_version, version))
                return
            with self._model_lock:
                retriever_object = self.retriever.get_objects()
                model.add_requests(retriever_object)

        return

    @log_exceptions
    def _retriever_thread(self) -> None:
        try:
            if self._has_initial_examples:
                self._initial_model_event.wait()

            while not self._abort_event.is_set():
                self._objects_for_model_version()
        except Exception as e:
            logger.error(e)
            self.stop()

    @log_exceptions
    def _infer_thread(self) -> None:
        try:
            if self._has_initial_examples:
                self._initial_model_event.wait()

            while not self._abort_event.is_set():
                result = self._model.get_results()
                self.selector.add_result(result)
        except Exception as e:
            logger.error(e)
            self.stop()

    @log_exceptions
    def _get_results(self, pipe) -> None:
        try:
            while True: # not self._abort_event.is_set():
                result = self.selector.get_result()
                if result is None:
                    break
                tile = SendTiles(
                    objectId=result.id, 
                    scoutIndex=self._scout_index,
                    score=result.score,
                    version=result.model_version,
                    attributes=result.attributes.get(),
                )
                pipe.send(tile.SerializeToString()) 
        except Exception as e:
            logger.error(e)
            self.stop()
            # raise e

    @log_exceptions
    def _get_labels(self, pipe) -> None:
        try:
            while not self._abort_event.is_set():
                try:
                    msg = pipe.recv()
                except EOFError as e:
                    continue
                request = LabelWrapper()
                request.ParseFromString(msg)
                self.distribute_label(request)
                if isinstance(self.selector, TokenSelector):
                    self.selector.receive_token_message(request)
        except Exception as e:
            logger.error(e)
            self.stop()

    @log_exceptions
    def _s2s_process(self, input_q, output_q) -> None:
        try:
            while not self._abort_event.is_set():
                (method, msg) = input_q.get()
                output = self.s2s_methods[method](msg)
                output_q.put(output)
        except Exception as e:
            logger.error(e)
            self.stop()

    @log_exceptions
    def _train_thread(self) -> None:
        while not self._abort_event.is_set():
            try:
                self._model_event.wait()
                self._train_model()
                self._model_event.clear()
                time.sleep(5)
            except Exception as e:
                logger.exception(e)

    def stop_model(self) -> None:
        with self._model_lock:
            logger.info("Stopping model")
            if self._model and self._model.is_running():
                self._model.stop()
        
    def _train_model(self) -> None:
        
        if self.trainer is None:
            return
        
        train_start = time.time()

        with self._model_lock:
            model = self._model

        if model is None and self.check_initial_model():
            logger.info("Loading initial model")
            model = self.trainer.load_model(content=self.initial_model.content)
            self._set_model(model, False)
            if self.enable_logfile:
                self.log_file.write("{:.3f} {}_{} Initial Model SET\n".format(time.time() - self.start_time,
                                                                              self.host_name, model.version))
            return
        
        with self._data_manager.get_examples(DatasetSplit.TRAIN) as train_dir:
            logger.info("Train dir {}".format(train_dir))
            model = self.trainer.train_model(train_dir)

        eval_start = time.time()
        logger.info('Trained model in {:.3f} seconds'.format(eval_start - train_start))
        if model is not None and self.enable_logfile:
            self.log_file.write("{:.3f} {}_{} TRAIN NEW MODEL in {} seconds\n".format( \
                    time.time() - self.start_time, self.host_name, model.version, eval_start - train_start))
        self._set_model(model, False)
        logger.info('Evaluated model in {:.3f} seconds'.format(time.time() - eval_start))

    def _set_model(self, model: Model, should_stage: bool) -> None:
        """Evaluates the trained model on the test data. If distributed waits
        for results from all scouts.  Based on condition promotes the trained
        model to the current version.  If ``only_use_better_models` then check
        the AUC scores to decide.

        Args:
            model (Model): Trained model
            should stage (bool): True, if the model needs to be shared

        """
        if self._abort_event.is_set():
            return

        model_stats = TestResults(version=model.version)

        with self._model_lock:
            should_notify = self._model is None
            if self._model and self._model.is_running():
                logger.info("Stopping model")
                self._model.stop()    
            self._model = model
            logger.info("Promoted New Model Version {}".format(self._model.version))
            if self.enable_logfile:
                self.log_file.write("{:.3f} {} {} Promoted New Model Version {}\n".format( \
                    time.time() - self.start_time, time.ctime(), self.host_name, model.version))
            self._model_stats = model_stats
            self._last_trained_version = model.version
        
        self.selector.new_model(model)

        if should_notify:
            self._initial_model_event.set()

    def _score_and_set_model(self, model: Model, should_stage: bool) -> None:
        """Evaluates the trained model on the test data. If distributed waits
        for results from all scouts.  Based on condition promotes the trained
        model to the current version.  If ``only_use_better_models` then check
        the AUC scores to decide.

        Args:
            model (Model): Trained model
            should stage (bool): True, if the model needs to be shared

        """
        logger.info("SCORE & SET MODEL")

        model_stats = None

        model_stats = TestResults(version=model.version)

        with self._model_lock:
            should_notify = self._model is None
            if self._model and self._model.is_running():
                self._model.stop()    
            self._model = model
            logger.info("Promoted New Model Version {}".format(self._model.version))
            self._model_stats = model_stats

            self._last_trained_version = model.version

        self.selector.new_model(model)
        if self.enable_logfile:
            self.log_file.write("{:.3f} {} Promoted New Model Version {}\n".format(
                time.time() - self.start_time, self.host_name, model.version))

        if should_notify:
            self._initial_model_event.set()


    @log_exceptions
    def _train_model_slave_thread(self) -> None:
        logger.info('Executing train request')
        with self._model_lock:
            model = self._model

        with self._data_manager.get_examples(DatasetSplit.TRAIN) as train_dir:
            train_start = time.time()
            model = self.trainer.train_model(train_dir)
            logger.info('Trained model in {:.3f} seconds'.format(time.time() - train_start))

        if not self.trainer.should_sync_model:
            with self._staged_model_condition:
                self._staged_models[model.version] = model
                self._staged_model_condition.notify_all()

    def _get_staging_model(self, model_version: int) -> Model:
        while not self._abort_event.is_set():
            with self._staged_model_condition:
                if model_version in self._staged_models:
                    model = self._staged_models[model_version]
                    break
                else:
                    logger.info('Waiting for model version {}'.format(model_version))
                self._staged_model_condition.wait()
        return model

