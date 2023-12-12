# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import io
import multiprocessing as mp
import os
import sys
import threading
import time
import zipfile
from collections import defaultdict
from multiprocessing.connection import _ConnectionBase
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from logzero import logger

from ...proto.messages_pb2 import (
    DatasetSplit,
    LabeledTile,
    LabelWrapper,
    MissionId,
    ModelArchive,
    SendTiles,
    TestResults,
)
from ..api.h2c_api import H2CSubscriber
from ..api.s2h_api import S2HPublisher
from ..api.s2s_api import S2SServicer, s2s_receive_request
from ..context.data_manager_context import DataManagerContext
from ..context.model_trainer_context import ModelContext
from ..retrain.retrain_policy_base import RetrainPolicyBase
from ..retrain.sampleInterval_policy import SampleIntervalPolicy
from ..retrieval.retriever import Retriever
from ..selection.selector_base import Selector
from ..selection.token_selector import TokenSelector
from .data_manager import DataManager
from .hawk_stub import HawkStub
from .model import Model
from .model_trainer import ModelTrainer
from .utils import get_server_ids, log_exceptions


class Mission(DataManagerContext, ModelContext):
    def __init__(
        self,
        mission_id: MissionId,
        scout_index: int,
        scouts: List[HawkStub],
        home_ip: str,
        retrain_policy: RetrainPolicyBase,
        root_dir: Path,
        port: int,
        retriever: Retriever,
        selector: Selector,
        bootstrap_zip: bytes,
        initial_model: ModelArchive,
        validate: bool = False,
    ):
        super().__init__()
        logger.info("Initialization")
        self.start_time = time.time()
        self._id = mission_id
        self.positives = 0
        self.negatives = 0
        self.sample_sent_to_model = 0

        self._scout_index = scout_index
        self._scouts = scouts

        self._retrain_policy = retrain_policy
        self._data_dir = root_dir / "data"
        self._log_dir = root_dir / "tb"
        self._model_dir = root_dir / "model"
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(self._model_dir, exist_ok=True)
        self.host_name = (get_server_ids()[0]).split(".")[0]
        self.home_ip = home_ip
        self.log_file = open(self._log_dir / f"log-{self.host_name}.txt", "a")
        self.result_path = str(self._log_dir / f"sent-{self.host_name}.txt")
        self.enable_logfile = True
        self.trainer: Optional[ModelTrainer] = None
        self.trainer_type = None

        self._port = port

        self.retriever = retriever
        self.selector = selector
        self.bootstrap_zip = bootstrap_zip

        if isinstance(self._retrain_policy, SampleIntervalPolicy):
            self._retrain_policy.total_tiles = self.retriever.total_tiles

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

        self._abort_event = threading.Event()

        self._data_manager = DataManager(self)
        self.selector.add_context(self)
        self.retriever.add_context(self)
        self.object_count = 0

        self.stats_count = 0

        # setup_result_connections()
        logger.info("SETTING UP S2H API")
        s2h_conn, s2h_input = mp.Pipe(False)
        p = mp.Process(
            target=S2HPublisher.s2h_send_tiles, args=(self.home_ip, s2h_conn)
        )
        p.start()

        self._result_thread = threading.Thread(
            target=self._get_results, args=(s2h_input,), name="get-results"
        )

        logger.info("SETTING UP H2C API")
        h2c_output, h2c_input = mp.Pipe(False)
        p = mp.Process(target=H2CSubscriber.h2c_receive_labels, args=(h2c_input,))
        self._label_thread = threading.Thread(
            target=self._get_labels, args=(h2c_output,), name="get-labels"
        )
        p.start()
        logger.info(f"SETTING UP S2S Server {self._scout_index}")
        s2s_input: mp.Queue[bytes] = mp.Queue()
        s2s_output: mp.Queue[bytes] = mp.Queue()
        p = mp.Process(
            target=s2s_receive_request,
            args=(
                s2s_input,
                s2s_output,
            ),
        )
        self._s2s_thread = threading.Thread(
            target=self._s2s_process,
            args=(
                s2s_input,
                s2s_output,
            ),
            name="internal",
        )
        p.start()

        self.start_time = time.time()
        s2s_object = S2SServicer(self)
        self.s2s_methods = {
            k.encode("utf-8"): getattr(s2s_object, k)
            for k in dir(s2s_object)
            if callable(getattr(s2s_object, k)) and k.startswith("s2s_")
        }

    def setup_trainer(self, trainer: ModelTrainer) -> None:
        logger.info("Setting up trainer")
        self.trainer = trainer
        threading.Thread(target=self._train_thread, name="train-model").start()
        self._retrain_policy.reset()
        self._model_event.set()

        # Wait for intial model to be trained
        while not self._model:
            pass

    def check_initial_model(self) -> bool:
        if self.initial_model is None:
            return False
        return len(self.initial_model.content) != 0

    def store_labeled_tile(self, tile: LabeledTile) -> None:
        self._data_manager.store_labeled_tile(tile)

    def distribute_label(self, label: LabelWrapper) -> None:
        self._data_manager.distribute_label(label)

    def get_example(self, example_set, label: str, example: str) -> Path:
        return self._data_manager.get_example_path(example_set, label, example)

    def get_test_results(self) -> TestResults:
        # We assume that the master is the source of truth
        with self._model_lock:
            stats = self._model_stats
            version = self._model.version if self._model is not None else -1

        if stats is not None:
            return stats

        return TestResults(version=version)

    def load_model(
        self,
        model_path: Path,
        content: bytes = b"",
        model_version: int = -1,
    ) -> Model:
        logger.info("Loading model")
        assert model_path.exists() or len(content)
        return self.trainer.load_model(model_path, content, model_version)

    def import_model(
        self, model_path: Path, content: bytes, model_version: int
    ) -> None:
        model = self.load_model(model_path, content, model_version)
        self._set_model(model, False)

    def export_model(self) -> ModelArchive:
        assert self._scout_index == 0

        memory_file = io.BytesIO()
        with self._model_lock:
            assert self._model is not None
            model = self._model
            model_dump = model.serialize()
            model_version = model.version
            # train_examples = model.train_examples

        with zipfile.ZipFile(memory_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model", model_dump)

        memory_file.seek(0)
        return ModelArchive(content=memory_file.getvalue(), version=model_version)

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

    def check_create_test(self) -> bool:
        return self._validate

    def new_labels_callback(
        self, new_positives: int, new_negatives: int, retrain: bool = True
    ) -> None:
        if self._abort_event.is_set():
            return
        logger.info(
            "New labels call back has been called... "
            f"positives: {new_positives}, negatives: {new_negatives}"
        )
        end_t = time.time()

        if retrain:
            self.positives += new_positives
            self.negatives += new_negatives

        # add line here to only call the function below if the retrain policy
        # is not the sample interval policy or simply use an if statement and
        # feed the total number of retreived samples
        if not isinstance(self._retrain_policy, SampleIntervalPolicy):
            self._retrain_policy.update(new_positives, new_negatives)
        if self.enable_logfile:
            self.log(
                f"{time.ctime()} NEW Examples "
                f"Positives {self._retrain_policy.positives} "
                f"Negatives {self._retrain_policy.negatives}",
                end_t,
            )
            self.log_file.flush()

        if not retrain:
            return

        if not isinstance(self._retrain_policy, SampleIntervalPolicy):
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
            threading.Thread(target=self._retriever_thread, name="get-objects").start()
            threading.Thread(target=self._infer_thread, name="infer-results").start()
            self._result_thread.start()
            self._label_thread.start()
            self._s2s_thread.start()

            self.start_time = time.time()
        except Exception:
            self.retriever.stop()
            self.selector.clear()
            self.stop()

    def stop(self) -> None:
        if not self._abort_event.is_set():
            if self._model is not None and self.enable_logfile:
                self.log(f"{self._model.version} SEARCH STOPPED")

        # Stop Mission
        self.enable_logfile = False
        self._abort_event.set()
        self.retriever.stop()
        self.selector.clear()
        logger.info("Selector clear called in mission.py...")
        self.log_file.close()
        sys.exit(0)

    def log(self, msg: str, end_t: Optional[float] = None) -> None:
        if not self.enable_logfile:
            return
        mission_time = self.mission_time(end_t)
        self.log_file.write(f"{mission_time:.3f} {self.host_name} {msg}\n")

    def get_example_directory(self, example_set) -> Path:
        return self._data_manager.get_example_directory(example_set)

    def _objects_for_model_version(self) -> None:
        if self._abort_event.is_set():
            return

        with self._model_lock:
            model = self._model

        # no current model, loop and try again. We really should have a
        # condition variable here to avoid busy looping.
        if model is None:
            return

        starting_version = model.version
        logger.info(f"Starting evaluation with model version {starting_version}")

        while not self._abort_event.is_set():
            with self._model_lock:
                if self._model is not None and self._model.version != starting_version:
                    logger.info(
                        f"Done evaluating with model version {starting_version} "
                        f"(new version {self._model.version} available)"
                    )
                    return

                # pop single retriever object from retriever result queue
                retriever_object = self.retriever.get_objects()

                # put single retriever object into model inference request queue
                model.add_requests(retriever_object)

                if isinstance(self._retrain_policy, SampleIntervalPolicy):
                    self._retrain_policy.interval_samples_retrieved += 1

                    if self._retrain_policy.should_retrain():
                        logger.info("Reached Retrain sample policy...")
                        self._retrain_policy.reset()
                        self._model_event.set()

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
                items_processed = self.selector.add_result(result)
                if (
                    isinstance(self.selector, TokenSelector)
                    and self.retriever.total_tiles == items_processed
                ):
                    self.selector.select_tiles(self.selector._k)
                if items_processed > self.retriever.total_tiles - 200:
                    sent_to_model = self._model.get_request_count()
                    logger.info(
                        f"Items retrieved, {self.retriever._stats.retrieved_tiles}, "
                        f"sent to model: {sent_to_model}, "
                        f"total items processed: {items_processed}"
                    )
                # if total number selected == total number of tiles, then
                # call select_tiles one last time.
        except Exception as e:
            logger.error(e)
            self.stop()

    @log_exceptions
    def _get_results(self, pipe: _ConnectionBase) -> None:
        try:
            while True:  # not self._abort_event.is_set():
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
    def _get_labels(self, pipe: _ConnectionBase) -> None:
        try:
            while not self._abort_event.is_set():
                try:
                    msg = pipe.recv()
                except EOFError:
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
            if self._model is not None and self._model.is_running():
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
                self.log(f"{model.version} Initial Model SET")
            return

        with self._data_manager.get_examples(DatasetSplit.TRAIN) as train_dir:
            logger.info(f"Train dir {train_dir}")
            model = self.trainer.train_model(train_dir)

        eval_start = time.time()
        logger.info(f"Trained model in {eval_start - train_start:.3f} seconds")
        if model is not None and self.enable_logfile:
            self.log(
                f"{model.version} TRAIN NEW MODEL in {eval_start - train_start} seconds"
            )
        self._set_model(model, False)
        logger.info(f"Evaluated model in {time.time() - eval_start:.3f} seconds")

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
            if self._model is not None and self._model.is_running():
                logger.info("Stopping model")
                self._model.stop()
            self._model = model
            self._model_stats = model_stats
            self._last_trained_version = model.version

        logger.info(f"Promoted New Model Version {model.version}")
        self.log(f"{time.ctime()} Promoted New Model Version {model.version}")
        self.selector.new_model(model)

        if should_notify:
            self._initial_model_event.set()
