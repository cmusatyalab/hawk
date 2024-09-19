# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Admin to Scouts internal api calls
"""

import base64
import dataclasses
import gc
import glob
import io
import json
import os
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Dict

import torch
from google.protobuf import json_format
from logzero import logger
from PIL import Image

from ...proto import Empty
from ...proto.messages_pb2 import (
    ChangeDeploymentStatus,
    Dataset,
    ImportModel,
    MissionId,
    MissionResults,
    MissionStats,
    ReexaminationStrategyConfig,
    RetrainPolicyConfig,
    ScoutConfiguration,
    SelectiveConfig,
    TestResults,
)
from ..core.hawk_stub import HawkStub
from ..core.mission import Mission
from ..core.mission_manager import MissionManager
from ..core.model_trainer import ModelTrainer
from ..core.utils import log_exceptions
from ..reexamination.full_reexamination_strategy import FullReexaminationStrategy
from ..reexamination.no_reexamination_strategy import NoReexaminationStrategy
from ..reexamination.reexamination_strategy import ReexaminationStrategy
from ..reexamination.top_reexamination_strategy import TopReexaminationStrategy
from ..retrain.absolute_policy import AbsolutePolicy
from ..retrain.model_policy import ModelPolicy
from ..retrain.percentage_policy import PercentagePolicy
from ..retrain.retrain_policy_base import RetrainPolicyBase
from ..retrain.sampleInterval_policy import SampleIntervalPolicy
from ..retrieval.frame_retriever import FrameRetriever
from ..retrieval.network_retriever import NetworkRetriever
from ..retrieval.random_retriever import RandomRetriever
from ..retrieval.retriever import Retriever
from ..retrieval.tile_retriever import TileRetriever
from ..retrieval.video_retriever import VideoRetriever
from ..selection.diversity_selector import DiversitySelector
from ..selection.selector_base import Selector
from ..selection.threshold_selector import ThresholdSelector
from ..selection.token_selector import TokenSelector
from ..selection.topk_selector import TopKSelector
from ..trainer.dnn_classifier.trainer import DNNClassifierTrainer
from ..trainer.dnn_classifier_radar.trainer import DNNClassifierTrainerRadar
from ..trainer.fsl.trainer import FSLTrainer
from ..trainer.yolo.trainer import YOLOTrainer
from ..trainer.yolo_radar.trainer import YOLOTrainerRadar

MODEL_FORMATS = ["pt", "pth"]


class A2SAPI:
    """Admin to Scouts API Calls

    API calls from admin to scouts to configure missions, explicitly start /
    stop mission, and other control calls.
    Uses Request-Response messaging.
    The network is not constricted.

    Attributes
    ----------
    _port : int
        TCP port number
    _manager : MissionManager
        manages hawk mission (sets and clears)
    """

    def __init__(self, port: int):
        self._port = port
        self._manager = MissionManager()

    def a2s_configure_scout(self, msg: bytes) -> bytes:
        """API call to configure scouts before mission

        Args:
            msg (str): Serialized ScoutConfiguration message from home

        Returns:
            bytes: SUCCESS or ERROR message
        """
        try:
            request = ScoutConfiguration()
            request.ParseFromString(msg)

            self._a2s_configure_scout(request)

            logger.info("Configured Successfully")
            reply = "SUCCESS"
        except Exception as e:
            reply = f"ERROR: {e}"
        return reply.encode()

    def a2s_start_mission(self, _arg: bytes) -> bytes:
        """API call to start mission

        Returns:
            bytes: SUCCESS or ERROR message
        """
        try:
            self._a2s_start_mission()
            reply = "SUCCESS"
        except Exception as e:
            reply = f"ERROR: {e}"
        return reply.encode()

    def a2s_stop_mission(self, _arg: bytes) -> bytes:
        """API call to stop mission

        Returns:
            bytes: SUCCESS or ERROR message
        """
        try:
            self._a2s_stop_mission()
            reply = "SUCCESS"
        except Exception as e:
            reply = f"ERROR: {e}"
        return reply.encode()

    def a2s_sync_deploy_status(self, msg: bytes) -> bytes:
        """sync knowledge about deploy status between home and scout"""
        try:
            reply = self._a2s_sync_deploy_status(msg)
        except Exception as e:
            reply = f"ERROR: {e}"
        return reply.encode()

    def a2s_change_deploy_status(self, msg: bytes) -> bytes:
        """change depl status"""
        request = ChangeDeploymentStatus()
        request.ParseFromString(msg)
        try:
            self._a2s_change_deploy_status(request)
            reply = "SUCCESS"
        except Exception as e:
            reply = f"ERROR: {e}"
        return reply.encode()

    def _a2s_sync_deploy_status(self, msg: bytes) -> str:
        retriever = self._manager.get_mission().retriever

        retriever.active_scout_ratio = float(msg.decode())

        return retriever.current_deployment_mode

    def _a2s_change_deploy_status(self, msg: ChangeDeploymentStatus) -> None:
        retriever = self._manager.get_mission().retriever

        retriever.active_scout_ratio = msg.ActiveScoutRatio

        if msg.ActiveStatus:
            retriever.scml_active_mode = True
        else:
            retriever.scml_active_mode = False

    def a2s_get_mission_stats(self, _arg: bytes) -> bytes:
        """API call to send mission stats to HOME

        Returns:
            str: serialized MissionStats message
        """
        try:
            stats = self._a2s_get_mission_stats()
            reply = stats.SerializeToString()
        except Exception as e:
            reply = f"ERROR: {e}".encode()
        return reply

    def a2s_new_model(self, msg: bytes) -> bytes:
        """API call to import new model from HOME

        Args:
            request (str): serialized ImportModel message

        Returns:
            bytes: SUCCESS or ERROR message
        """
        try:
            request = ImportModel()
            request.ParseFromString(msg)

            self._a2s_new_model(request)

            reply = "SUCCESS"
        except Exception as e:
            reply = f"ERROR: {e}"
        return reply.encode()

    def a2s_get_test_results(self, msg: bytes) -> bytes:
        """API call to test the model on the TEST dataset

        Args:
            request (str): path to the TEST dataset on the scouts

        Returns:
            str: serialized MissionResults message
        """
        try:
            test_path = msg.decode()

            logger.info(f"Testing {test_path}")
            assert os.path.exists(test_path)

            results = self._a2s_get_test_results(test_path)
            reply = results.SerializeToString()
        except Exception as e:
            reply = f"ERROR: {e}".encode()
        return reply

    def a2s_get_post_mission_archive(self, _arg: bytes) -> bytes:
        """API call to send mission models and logs archive

        Returns:
            bytes: mission archive zip file as a byte array
        """
        try:
            reply = self._a2s_get_post_mission_archive()
        except Exception:
            reply = Empty
        return reply

    @log_exceptions
    def _a2s_configure_scout(self, request: ScoutConfiguration) -> None:
        """Function to parse config message and setup for mission

        Args:
            request (ScoutConfiguration): configuration message
        """
        root_dir = Path(request.missionDirectory) / "data"
        assert root_dir.is_dir(), f"Root directory {root_dir} does not exist"
        model_dir = root_dir / request.missionId / "model"

        mission_id = MissionId(value=request.missionId)

        this_host = request.scouts[request.scoutIndex]
        scouts = [HawkStub(scout, this_host) for scout in request.scouts]

        retriever = self._get_retriever(request.missionId, request.dataset, this_host)

        retrain_policy = self._get_retrain_policy(request.retrainPolicy, model_dir)
        if request.retrainPolicy.HasField("sample"):
            assert isinstance(retrain_policy, SampleIntervalPolicy)
            retrain_policy.num_interval_sample(retriever.total_tiles)

        reexamination_strategy = self._get_reexamination_strategy(
            request.reexamination, retriever
        )
        selector = self._get_selector(
            request.missionId, request.selector, reexamination_strategy
        )

        # Setting up Mission with config params
        logger.info("Start setting up mission")
        logger.info(
            f"Class list: {request.class_list}, {request.novel_class_discovery}"
        )
        mission = Mission(
            mission_id,
            request.scoutIndex,
            scouts,
            request.homeIP,
            retrain_policy,
            root_dir / mission_id.value,
            self._port,
            retriever,
            selector,
            request.bootstrapZip,
            request.initialModel,
            request.trainStrategy,
            list(request.class_list),
            dict(request.scml_deploy_opts.scout_dict),
            request.validate,
            request.novel_class_discovery,
            # add base model field for radar missions
            # add request.train_strategy here to be able to pass to data manager.
        )
        logger.info("Finished setting up mission")
        self._manager.set_mission(mission)

        # Setting up mission trainer
        model = request.trainStrategy
        if model.HasField("dnn_classifier_radar"):
            config = model.dnn_classifier_radar
            trainer: ModelTrainer = DNNClassifierTrainerRadar(
                mission, dict(config.args)
            )
        elif model.HasField("dnn_classifier"):
            config = model.dnn_classifier
            trainer = DNNClassifierTrainer(mission, dict(config.args))
        elif model.HasField("yolo"):
            config = model.yolo
            trainer = YOLOTrainer(mission, dict(config.args))
        elif model.HasField("yolo_radar"):
            config = model.yolo_radar
            trainer = YOLOTrainerRadar(mission, dict(config.args))
        elif model.HasField("fsl"):
            config = model.fsl
            support_path = config.args["support_path"]

            # Saving support image
            support_data = config.args["support_data"]
            data = base64.b64decode(support_data.encode("utf8"))
            image = Image.open(io.BytesIO(data))
            image.save(support_path)

            trainer = FSLTrainer(mission, dict(config.args))
        else:
            raise NotImplementedError(
                f"unknown model: {json_format.MessageToJson(model)}"
            )

        mission.setup_trainer(trainer)
        logger.info(f"Create mission with id {request.missionId}")

        # Constricting bandwidth
        # Only supports one bandwidth
        logger.info(request.bandwidthFunc)
        if not request.selector.HasField("token"):
            self._setup_bandwidth(request.bandwidthFunc[request.scoutIndex])
        if mission.enable_logfile:
            mission.log("SEARCH CREATED")

    def _setup_bandwidth(self, bandwidth_func: str) -> None:
        """Function for FireQos Bandwidth limiting"""
        bandwidth_map = {
            "100k": "/root/fireqos/scenario-100k.conf",
            "30k": "/root/fireqos/scenario-30k.conf",
            "12k": "/root/fireqos/scenario-12k.conf",
        }
        logger.info(bandwidth_func)
        bandwidth_list = json.loads(bandwidth_func)
        default_file = bandwidth_map["100k"]
        # bandwidth_file = default_file
        if bandwidth_list[0] == "0k":
            return

        for _time_stamp, bandwidth in bandwidth_list:
            bandwidth_file = bandwidth_map.get(bandwidth.lower(), default_file)

        # start fireqos
        bandwidth_cmd = ["fireqos", "start", str(bandwidth_file)]
        b = subprocess.Popen(bandwidth_cmd)
        b.communicate()

    @log_exceptions
    def _a2s_start_mission(self) -> None:
        """Function to start mission"""
        logger.info("Starting mission calling mission")
        mission = self._manager.get_mission()
        mission_id = mission.mission_id.value
        logger.info(f"Starting mission with id {mission_id}")
        mission.start()
        mission.log("SEARCH STARTED")

    @log_exceptions
    def _a2s_stop_mission(self) -> None:
        """Function to stop mission"""
        try:
            mission = self._manager.get_mission()
            mission_id = mission.mission_id.value
            logger.info(f"Stopping mission with id {mission_id}")
            mission.log("SEARCH STOPPED")
            mission.stop()
            self._manager.remove_mission()
        finally:
            # Stop fireqos
            bandwidth_cmd = ["fireqos", "stop"]
            b = subprocess.Popen(bandwidth_cmd)
            b.communicate()

    @log_exceptions
    def _a2s_get_mission_stats(self) -> MissionStats:
        """Function to send mission stats to home

        Returns:
            MissionStats
        """
        mission = self._manager.get_mission()
        time_now = mission.mission_time()
        mission.log("SEARCH STATS (collecting...)")

        logger.info("Before retriever get stats in a2s...")
        retriever_stats = mission.retriever.get_stats()
        selector_stats = mission.selector.get_stats()

        mission_stats = dataclasses.asdict(retriever_stats)
        mission_stats.update(dataclasses.asdict(selector_stats))
        keys_to_remove = [
            "total_objects",
            "processed_objects",
            "dropped_objects",
            "false_negatives",
        ]
        for k in list(mission_stats):
            v = mission_stats[k]
            mission_stats[k] = str(v)
            if k in keys_to_remove:
                del mission_stats[k]

        model_version = mission._model.version if mission._model is not None else -1

        mission_stats.update(
            {
                "server_time": str(time_now),
                "version": str(model_version),
                "msg": "stats",
                "ctime": str(time.ctime()),
                "server_positives": str(mission.positives),
                "server_negatives": str(mission.negatives),
            }
        )

        reply = MissionStats(
            totalObjects=retriever_stats.total_objects,
            processedObjects=selector_stats.processed_objects,
            droppedObjects=retriever_stats.dropped_objects,
            falseNegatives=0,
            others=mission_stats,
        )
        if mission.enable_logfile:
            mission.log("SEARCH STATS")
        return reply

    @log_exceptions
    def _a2s_new_model(self, request: ImportModel) -> None:
        """Function to import new model from HOME

        Args:
            request (ImportModel) ImportModel message
        """
        mission = self._manager.get_mission()
        model = request.model
        path = Path(request.path)
        version = model.version
        mission.import_model(path, model.content, version)
        logger.info("[IMPORT] FINISHED Model Import")
        mission.log("IMPORT MODEL")

    @log_exceptions
    def _a2s_get_test_results(self, request: str) -> MissionResults:
        """Function to test the model on the TEST dataset

        Args:
            request (str): path to the TEST dataset on the scouts

        Returns:
            MissionResults
        """
        test_path = Path(request)
        if not test_path.is_file():
            raise Exception(f"{test_path} does not exist")

        mission = self._manager.get_mission()
        model_dir = str(mission.model_dir)
        files = sorted(glob.glob(os.path.join(model_dir, "*.*")))
        model_paths = [x for x in files if x.split(".")[-1].lower() in MODEL_FORMATS]
        logger.info(model_paths)

        def get_version(path: Path, idx: int) -> int:
            name = path.name
            try:
                version = int(name.split("model-")[-1].split(".")[0])
            except Exception:
                version = idx
            return version

        results: Dict[int, TestResults] = {}
        for idx, filename in enumerate(model_paths):
            path = Path(filename)
            version = get_version(path, idx)
            logger.info(f"model {path} version {version}")
            # create trainer and check
            model = mission.load_model(path, model_version=version)
            result = model.evaluate_model(test_path)
            results[version] = result

        return MissionResults(results=results)

    @log_exceptions
    def _a2s_get_post_mission_archive(self) -> bytes:
        """Function to send mission models and logs archive

        Returns:
            bytes: mission archive zip file as a byte array
        """
        mission = self._manager.get_mission()
        # data_dir = mission.data_dir
        model_dir = mission.model_dir

        mission_archive = io.BytesIO()
        with zipfile.ZipFile(
            mission_archive, "w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            for dirname, _subdirs, files in os.walk(model_dir):
                zf.write(dirname)
                for filename in files:
                    zf.write(os.path.join(dirname, filename))

        logger.info("[IMPORT] FINISHED Archiving Mission models")
        return mission_archive.getbuffer()

    def _get_retrain_policy(
        self, retrain_policy: RetrainPolicyConfig, model_dir: Path
    ) -> RetrainPolicyBase:
        if retrain_policy.HasField("absolute"):
            return AbsolutePolicy(
                retrain_policy.absolute.threshold, retrain_policy.absolute.onlyPositives
            )
        elif retrain_policy.HasField("percentage"):
            return PercentagePolicy(
                retrain_policy.percentage.threshold,
                retrain_policy.percentage.onlyPositives,
            )
        elif retrain_policy.HasField("model"):
            logger.info("Model Policy")
            return ModelPolicy(str(model_dir))
        elif retrain_policy.HasField("sample"):
            return SampleIntervalPolicy(retrain_policy.sample.num_intervals)
        else:
            raise NotImplementedError(
                "unknown retrain policy: {}".format(
                    json_format.MessageToJson(retrain_policy)
                )
            )

    def _get_selector(
        self,
        mission_id: str,
        selector: SelectiveConfig,
        reexamination_strategy: ReexaminationStrategy,
    ) -> Selector:
        if selector.HasField("topk"):
            top_k_param = json_format.MessageToDict(selector.topk)
            logger.info(f"TopK Params {top_k_param}")
            return TopKSelector(
                mission_id,
                selector.topk.k,
                selector.topk.batchSize,
                selector.topk.countermeasure_threshold,
                selector.topk.total_countermeasures,
                reexamination_strategy,
            )
        elif selector.HasField("token"):
            return TokenSelector(
                mission_id,
                selector.token.initial_samples,
                selector.token.batch_size,
                selector.token.countermeasure_threshold,
                selector.token.total_countermeasures,
                reexamination_strategy,
            )
        elif selector.HasField("threshold"):
            return ThresholdSelector(
                mission_id,
                selector.threshold.threshold,
                reexamination_strategy,
            )
        elif selector.HasField("diversity"):
            top_k_param = json_format.MessageToDict(selector.diversity)
            logger.info(f"Diversity Params {top_k_param}")
            return DiversitySelector(
                mission_id,
                selector.diversity.k,
                selector.diversity.batchSize,
                selector.diversity.countermeasure_threshold,
                selector.diversity.total_countermeasures,
                reexamination_strategy,
            )
        else:
            raise NotImplementedError(
                f"unknown selector: {json_format.MessageToJson(selector)}"
            )

    def _get_reexamination_strategy(
        self,
        reexamination_strategy: ReexaminationStrategyConfig,
        retriever: Retriever,
    ) -> ReexaminationStrategy:
        reexamination_type = reexamination_strategy.type
        if reexamination_type == "none":
            return NoReexaminationStrategy(retriever)
        elif reexamination_type == "top":
            return TopReexaminationStrategy(retriever, reexamination_strategy.k)
        elif reexamination_type == "full":
            return FullReexaminationStrategy(retriever)
        else:
            raise NotImplementedError(
                "unknown reexamination strategy: {}".format(
                    json_format.MessageToJson(reexamination_strategy)
                )
            )

    def _get_retriever(
        self, mission_id: str, dataset: Dataset, this_host: str
    ) -> Retriever:
        if dataset.HasField("tile"):
            return TileRetriever(mission_id, dataset.tile)
        elif dataset.HasField("frame"):
            return FrameRetriever(mission_id, dataset.frame)
        elif dataset.HasField("random"):
            return RandomRetriever(mission_id, dataset.random)
        elif dataset.HasField("video"):
            return VideoRetriever(mission_id, dataset.video)
        elif dataset.HasField("network"):
            return NetworkRetriever(mission_id, dataset.network, this_host)
        else:
            raise NotImplementedError(
                f"unknown dataset: {json_format.MessageToJson(dataset)}"
            )
