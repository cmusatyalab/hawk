# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import io
import multiprocessing as mp
import os
import threading
import time
import zipfile
from collections import defaultdict
from multiprocessing.connection import _ConnectionBase
from pathlib import Path
from typing import TYPE_CHECKING

from logzero import logger

from ...classes import ClassCounter, ClassName, class_name_to_str
from ...proto.messages_pb2 import (
    BoundingBox,
    DatasetSplit,
    LabeledTile,
    MissionId,
    ModelArchive,
    SendLabel,
    SendTile,
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
from ..stats import (
    HAWK_MISSION_CONFIGURING,
    HAWK_MISSION_RUNNING,
    HAWK_MISSION_TRAINING_BOOTSTRAP,
    HAWK_MISSION_WAITING,
    HAWK_MODEL_REEXAMINING,
    HAWK_MODEL_TRAINING,
    HAWK_MODEL_VERSION,
)
from ..trainer import novel_class_discover
from .data_manager import DataManager
from .hawk_stub import HawkStub
from .model import Model
from .model_trainer import ModelTrainer
from .result_provider import ResultProvider
from .utils import get_server_ids, log_exceptions

if TYPE_CHECKING:
    from ...proto.messages_pb2 import DatasetSplitValue, TrainConfig


class Mission(DataManagerContext, ModelContext):
    def __init__(
        self,
        mission_id: MissionId,
        scout_index: int,
        scouts: list[HawkStub],
        home_ip: str,
        retrain_policy: RetrainPolicyBase,
        root_dir: Path,
        port: int,
        retriever: Retriever,
        selector: Selector,
        bootstrap_zip: bytes,
        initial_model: ModelArchive,
        base_model: ModelArchive,
        train_strategy: TrainConfig,
        class_list: list[str],
        scml_deploy_options: dict[str, int],
        validate: bool = False,
        novel_class_discovery: bool = False,
        sub_class_discovery: bool = False,
    ):
        super().__init__(retriever=retriever)

        self.start_time = time.time()

        self._mission_configuring = HAWK_MISSION_CONFIGURING.labels(
            mission=mission_id.value
        )
        self._mission_training_bootstrap = HAWK_MISSION_TRAINING_BOOTSTRAP.labels(
            mission=mission_id.value
        )
        self._mission_waiting = HAWK_MISSION_WAITING.labels(mission=mission_id.value)
        self._mission_running = HAWK_MISSION_RUNNING.labels(mission=mission_id.value)

        # switch state from "idle/stopped" to "configuring"
        self._mission_configuring.set(1)

        logger.info("Initialization")
        self._id = mission_id
        self.positives = 0
        self.negatives = 0
        self.sample_sent_to_model = 0

        self._scout_index = scout_index
        self._scouts = scouts

        self._retrain_policy = retrain_policy
        self._data_dir = root_dir / "data"
        self._feature_vector_dir = self._data_dir / "feature_vectors"
        self._log_dir = root_dir / "tb"
        self._model_dir = root_dir / "model"
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(self._model_dir, exist_ok=True)
        os.makedirs(self._feature_vector_dir, exist_ok=True)
        self.host_name = (get_server_ids()[0]).split(".")[0]
        self.home_ip = home_ip
        self.log_file = self._log_dir.joinpath(f"log-{self.host_name}.txt").open("a")
        self.enable_logfile = True
        self.trainer: ModelTrainer | None = None
        self.trainer_type = None

        self._port = port

        self.retriever = retriever
        self.selector = selector
        self.bootstrap_zip = bootstrap_zip
        self.train_strategy = train_strategy

        if isinstance(self._retrain_policy, SampleIntervalPolicy):
            self._retrain_policy.total_tiles = self.retriever.total_tiles

        self.initial_model = initial_model
        self.base_model = base_model
        if self.base_model is not None:
            base_model_path = self._model_dir / "base_model.pth"
            with open(base_model_path, "wb") as f:
                f.write(self.base_model.content)
            logger.info(f" base model path: {base_model_path}")
        self._validate = validate
        self.scml_deploy_options = scml_deploy_options

        # Indicates that the mission will seed the strategy with an initial set
        # of examples. The strategy should therefore hold off on returning
        # inference results until its underlying model is trained
        self._has_initial_examples = True

        self._model: Model | None = None

        self._model_training = HAWK_MODEL_TRAINING.labels(mission=mission_id.value)
        self._model_reexamining = HAWK_MODEL_REEXAMINING.labels(
            mission=mission_id.value
        )
        self._model_version = HAWK_MODEL_VERSION.labels(mission=mission_id.value)
        self._model_version.set(-1)

        self._model_stats = TestResults(version=-1)
        self._model_lock = threading.Lock()
        self._initial_model_event = threading.Event()
        self._model_event = threading.Event()
        self._object_model_excess = 2000

        self._test_results: dict[int, list[list[tuple[int, float]]]] = defaultdict(list)
        self._results_condition = mp.Condition()

        self._abort_event = threading.Event()

        logger.info("Initializing Class manager and class objects:")
        self.class_list.extend(ClassName(name) for name in class_list)
        logger.info(f"Class list in mission: {self.class_list}")

        self.novel_class_discovery = novel_class_discovery
        self.sub_class_discovery = sub_class_discovery
        logger.info(f"Novel Class discovery: {self.novel_class_discovery}")

        ## create class objects here.
        self._data_manager = DataManager(self)
        self.selector.add_context(self)
        self.retriever.add_context(self)

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
        h2c_port = port + 1
        # receive labels continuously over ZMQ socket
        p = mp.Process(
            target=H2CSubscriber.h2c_receive_labels, args=(h2c_input, h2c_port)
        )
        self._label_thread = threading.Thread(
            target=self._get_labels, args=(h2c_output,), name="get-labels"
        )
        p.start()
        logger.info(f"SETTING UP S2S Server {self._scout_index}")
        s2s_input: mp.Queue[bytes] = mp.Queue()
        s2s_output: mp.Queue[bytes] = mp.Queue()
        s2s_port = port + 2
        p = mp.Process(
            target=s2s_receive_request,
            args=(
                s2s_input,
                s2s_output,
                s2s_port,
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

        ## setup clustering process, if needed
        if self.novel_class_discovery:
            self.clustering_input_queue: mp.Queue[ResultProvider] = mp.Queue()
            self.labels_queue: mp.Queue[tuple[str, Path]] = mp.Queue()
            novel_cluster_process = mp.Process(
                target=novel_class_discover.main,
                args=(
                    self.retriever,
                    self.clustering_input_queue,
                    self.labels_queue,
                    s2h_input,
                    self._feature_vector_dir,
                    self._scout_index,
                ),
            )
            novel_cluster_process.start()

    def setup_trainer(self, trainer: ModelTrainer) -> None:
        logger.info("Setting up trainer")
        self.trainer = trainer
        threading.Thread(target=self._train_thread, name="train-model").start()
        self._retrain_policy.reset()
        self._model_event.set()

        # indicate we're blocked waiting for the initial model to be trained
        self._mission_training_bootstrap.set(1)

        # Wait for initial model to be trained
        while not self._model:
            time.sleep(0.5)

        self._mission_training_bootstrap.set(0)

    def check_initial_model(self) -> bool:
        if self.initial_model is None:
            return False
        return len(self.initial_model.content) != 0

    def store_labeled_tile(self, tile: LabeledTile, net: bool = False) -> None:
        self._data_manager.store_labeled_tile(tile, net)

    def distribute_label(self, label: SendLabel) -> None:
        self._data_manager.distribute_label(label)

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
        assert self.trainer is not None
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
            self._model_version.set(-1)
            self._model_stats = TestResults(version=-1)

        self.selector.new_model(None)

    @property
    def scout_index(self) -> int:
        return self._scout_index

    @property
    def scouts(self) -> list[HawkStub]:
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
        self,
        sample_counts: ClassCounter,
        retrain: bool = True,
    ) -> None:
        if self._abort_event.is_set():
            return

        new_positives = sample_counts.positives
        new_negatives = sample_counts.negatives

        logger.info(
            "New labels call back has been called... "
            f"{new_positives=}, {new_negatives=}, by class={sample_counts!r}"
        )
        end_t = time.time()

        if retrain:
            self.positives += new_positives
            self.negatives += new_negatives
            ## add new samples list for multiclass, if necessary

        # add line here to only call the function below if the retrain policy
        # is not the sample interval policy or simply use an if statement and
        # feed the total number of retrieved samples
        if not isinstance(self._retrain_policy, SampleIntervalPolicy):
            # add new pos and neg to current tally
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
            # if self._retrain_policy.should_retrain():
            #     should_retrain = True
            # else:
            #     with self._model_lock:
            #        should_retrain = self._model is None

            should_retrain = self._retrain_policy.should_retrain()

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

            # switch state from "waiting for start" to "running"
            self._mission_running.set(1)
            self._mission_waiting.set(0)

            self.start_time = time.time()
        except Exception:
            self.retriever.stop()
            self.selector.clear()
            self.stop()

    def stop(self) -> None:
        if (
            not self._abort_event.is_set()
            and self._model is not None
            and self.enable_logfile
        ):
            self.log(f"{self._model.version} SEARCH STOPPED")

        # Stop Mission
        self.enable_logfile = False
        self._abort_event.set()
        self.retriever.stop()
        self.selector.clear()
        logger.info("Selector clear called in mission.py...")
        self.log_file.close()

        # switch state from "running" to "stopped"
        self._mission_running.set(0)

    def log(self, msg: str, end_t: float | None = None) -> None:
        if not self.enable_logfile:
            return
        mission_time = self.mission_time(end_t)
        self.log_file.write(f"{mission_time:.3f} {self.host_name} {msg}\n")

    def get_example_directory(self, example_set: DatasetSplitValue) -> Path:
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
            # we should do better, but it requires untangling the various
            # queues and moving get_ml_batch into the inference loop.
            # for now do a blocking get for the next object.
            object_id = self.retriever._get_objectid()
            if object_id is None:
                break

            with self._model_lock:
                if self._model is not None and self._model.version != starting_version:
                    logger.info(
                        f"Done evaluating with model version {starting_version} "
                        f"(new version {self._model.version} available)"
                    )
                    ## make sure to put this back in retriever put object
                    self.retriever.put_objectid(object_id, dup=True)
                    logger.info(
                        "\n\nATTENTION --- PUTTING OBJECT BACK  IN RETRIEVER QUEUE\n\n"
                    )
                    return

                # pop single retriever object from retriever result queue

                # will need to add put_objects(retriever_object) when we move
                # get_objects() outside the lock above.

                # put single retriever object into model inference request queue
                model.add_requests(object_id)

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
            assert self._model is not None

            while not self._abort_event.is_set():
                result = self._model.get_results()
                if self.novel_class_discovery and result is not None:
                    ## put fv in queue for novel class clustering.
                    self.clustering_input_queue.put(result)
                # items_processed = \
                self.selector.add_result(result)
                ## line here to send result object to clustering process
                ## write attributes from result object to a file

                ## add final batch of samples for transmission once all tiles
                ## have been retrieved.
                # if (
                #    isinstance(self.selector, TokenSelector)
                #    and self.retriever.total_tiles == items_processed
                # ):
                #    self.selector.select_tiles(self.selector._k)
                # if total number selected == total number of tiles, then
                # call select_tiles one last time.
        except Exception as e:
            logger.error(e)
            self.stop()

    @log_exceptions
    def _get_results(self, pipe: _ConnectionBase) -> None:
        try:
            while True:  # not self._abort_event.is_set():
                ## if  in scml idle mode, time.sleep(10), and continue
                result = self.selector.get_result()
                if result is None:
                    break

                bboxes = [
                    BoundingBox(
                        x=bbox.get("x", 0.5),
                        y=bbox.get("y", 0.5),
                        w=bbox.get("w", 1.0),
                        h=bbox.get("h", 1.0),
                        class_name=class_name_to_str(bbox["class_name"]),
                        confidence=bbox["confidence"],
                    )
                    for bbox in result.bboxes
                ]

                oracle_data = self.retriever.get_oracle_data(result.object_id)

                tile = SendTile(
                    _objectId=result.object_id.serialize_oid(),
                    scoutIndex=self._scout_index,
                    version=result.model_version,
                    feature_vector=result.feature_vector,
                    oracle_data=[obj.to_protobuf() for obj in oracle_data],
                    boundingBoxes=bboxes,
                    novel_sample=False,
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
                request = SendLabel()
                request.ParseFromString(msg)
                self.distribute_label(request)
                if isinstance(self.selector, TokenSelector):  ## and scout is active
                    self.selector.receive_token_message(request)
                    # need a spin loop here when scout is idle.  this will will
                    # prevent further samples from being sent to home when
                    # idle. But need to be able to break out of loop is scout
                    # mode set back to Active.
                    # not yet implemented.  May not need this feature.

        except Exception as e:
            logger.error(e)
            self.stop()

    @log_exceptions
    def _s2s_process(
        self, input_q: mp.Queue[tuple[bytes, bytes]], output_q: mp.Queue[bytes]
    ) -> None:
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
                self._model_training.set(1)

                self._train_model()

                self._model_training.set(0)
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
        with self._data_manager.get_examples(
            DatasetSplit.TRAIN
        ) as train_dir:  # important
            logger.info(f"Train dir {train_dir}")
            self.selector.add_easy_negatives(train_dir)
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
            self._model_version.set(model.version)
            self._model_stats = model_stats
        logger.info(f"Promoted New Model Version {model.version}")
        self.log(f"{time.ctime()} Promoted New Model Version {model.version}")

        # This ultimately calls the reexamination inference through the new
        # model after training completed
        self._model_reexamining.set(1)
        self.selector.new_model(model)
        self._model_reexamining.set(0)

        if should_notify:
            self._initial_model_event.set()

    @property
    def model_version(self) -> int:
        return self._model.version if self._model is not None else -1
