# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Admin to Scouts internal api calls
"""

import base64
import copy
import gc
import glob
import io
import json
import os
import subprocess
import time
import zipfile
from pathlib import Path

import torch
from google.protobuf import json_format
from logzero import logger
from PIL import Image

from ...ports import S2S_PORT
from ...proto import Empty
from ...proto.messages_pb2 import (
    Dataset,
    ImportModel,
    MissionId,
    MissionResults,
    MissionStats,
    ReexaminationStrategyConfig,
    RetrainPolicyConfig,
    ScoutConfiguration,
    SelectiveConfig,
)
from ..core.hawk_stub import HawkStub
from ..core.mission import Mission
from ..core.mission_manager import MissionManager
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
from ..trainer.fsl.trainer import FSLTrainer
from ..trainer.yolo.trainer import YOLOTrainer

MODEL_FORMATS = ["pt", "pth"]


class A2SAPI:
    """Admin to Scouts API Calls

    API calls from admin to scouts to configure missions, explicitly start / stop mission,
    and other control calls. Uses Request-Response messaging. The network is not constricted.

    Attributes
    ----------
    _port : int
        TCP port number
    _manager : MissionManager
        manages hawk mission (sets and clears)
    _trainer : ModelTrainerBase
        TrainingStrategy used in mission
    """

    def __init__(self, port: int):
        self._port = port
        self._manager = MissionManager()
        self._trainer = None

    @log_exceptions
    def a2s_configure_scout(self, msg: str):
        """API call to configure scouts before mission

        Args:
            msg (str): Serialized ScoutConfiguration message from home

        Returns:
            bytes: SUCESS or ERROR message
        """
        try:
            request = ScoutConfiguration()
            request.ParseFromString(msg)

            reply = self._a2s_configure_scout(request)
            logger.info("Configured Successfully")
        except Exception as e:
            logger.exception(e)
            reply = (f"ERROR: {e}").encode()
            # raise e

        return reply

    @log_exceptions
    def a2s_start_mission(self, _arg):
        """API call to start mission

        Returns:
            bytes: SUCESS or ERROR message
        """
        try:
            reply = self._a2s_start_mission()
            return reply
        except Exception as e:
            logger.exception(e)
            raise e

    @log_exceptions
    def a2s_stop_mission(self, _arg):
        """API call to stop mission

        Returns:
            bytes: SUCESS or ERROR message
        """
        try:
            reply = self._a2s_stop_mission()
            return reply
        except Exception as e:
            logger.exception(e)
            raise e

    @log_exceptions
    def a2s_get_mission_stats(self, _arg):
        """API call to send mission stats to HOME

        Returns:
            str: serialized MissionStats message
        """
        try:
            reply = self._a2s_get_mission_stats()
            return reply
        except Exception as e:
            logger.exception(e)
            raise e

    @log_exceptions
    def a2s_new_model(self, msg: str):
        """API call to import new model from HOME

        Args:
            request (str): serialized ImportModel message

        Returns:
            bytes: SUCESS or ERROR message
        """
        try:
            request = ImportModel()
            request.ParseFromString(msg)
            reply = self._a2s_new_model(request)
            return reply
        except Exception as e:
            logger.exception(e)
            raise e

    @log_exceptions
    def a2s_get_test_results(self, msg: str):
        """API call to test the model on the TEST dataset

        Args:
            request (str): path to the TEST dataset on the scouts

        Returns:
            str: serialized MissionResults message
        """
        try:
            test_path = msg.decode("utf-8")
            logger.info(f"Testing {test_path}")
            assert os.path.exists(test_path)
            reply = self._a2s_get_test_results(test_path)
            return reply
        except Exception as e:
            logger.exception(e)
            raise e

    @log_exceptions
    def a2s_get_post_mission_archive(self, _arg):
        """API call to send mission models and logs archive

        Returns:
            bytes: mission archive zip file as a byte array
        """
        try:
            reply = self._a2s_get_post_mission_archive()
        except Exception as e:
            logger.exception(e)
            reply = Empty
        return reply

    @log_exceptions
    def _a2s_configure_scout(self, request: ScoutConfiguration):
        """Function to parse config message and setup for mission

        Args:
            request (ScoutConfiguration): configuration message

        Returns:
            bytes: SUCESS or ERROR message

        """
        try:
            root_dir = Path(request.missionDirectory) / "data"
            assert root_dir.is_dir(), f"Root directory {root_dir} does not exist"
            model_dir = root_dir / request.missionId / "model"

            mission_id = MissionId(value=request.missionId)
            retrain_policy = self._get_retrain_policy(request.retrainPolicy, model_dir)
            retriever = self._get_retriever(request.dataset)
            if request.retrainPolicy.HasField("sample"):
                retrain_policy.num_interval_sample(retriever.total_tiles)
            this_host = request.scouts[request.scoutIndex]
            scouts = [HawkStub(scout, S2S_PORT, this_host) for scout in request.scouts]

            # Setting up Mission with config params
            logger.info("Start setting up mission")
            mission = Mission(
                mission_id,
                request.scoutIndex,
                scouts,
                request.homeIP,
                retrain_policy,
                root_dir / mission_id.value,
                self._port,
                retriever,
                self._get_selector(request.selector, request.reexamination),
                request.bootstrapZip,
                request.initialModel,
                request.validate,
            )
            logger.info("Finished setting up mission")
            self._manager.set_mission(mission)

            # Setting up mission trainer
            model = request.trainStrategy
            trainer = None
            if model.HasField("dnn_classifier"):
                config = model.dnn_classifier
                trainer = DNNClassifierTrainer(mission, config.args)
            elif model.HasField("yolo"):
                config = model.yolo
                trainer = YOLOTrainer(mission, config.args)
            elif model.HasField("fsl"):
                config = model.fsl
                support_path = config.args["support_path"]

                # Saving support image
                support_data = config.args["support_data"]
                data = base64.b64decode(support_data.encode("utf8"))
                image = Image.open(io.BytesIO(data))
                image.save(support_path)

                trainer = FSLTrainer(mission, config.args)
            else:
                raise NotImplementedError(
                    f"unknown model: {json_format.MessageToJson(model)}"
                )

            self._trainer = trainer
            mission.setup_trainer(trainer)
            logger.info(f"Create mission with id {request.missionId}")

            # Constricting bandwidth
            # Only supports one bandwidth
            logger.info(request.bandwidthFunc)
            if not request.selector.HasField("token"):
                self._setup_bandwidth(request.bandwidthFunc[request.scoutIndex])
            if mission.enable_logfile:
                mission.log_file.write(
                    "{:.3f} {} SEARCH CREATED\n".format(
                        time.time() - mission.start_time, mission.host_name
                    )
                )

            reply = b"SUCCESS"
        except Exception as e:
            logger.exception("Error during setup")
            reply = f"ERROR: {e}".encode()
        return reply

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
        return

    def _a2s_start_mission(self):
        """Function to start mission

        Returns:
            bytes: SUCESS or ERROR message
        """
        try:
            logger.info("Starting mission calling mission")
            mission = self._manager.get_mission()
            mission_id = mission.mission_id.value
            logger.info(f"Starting mission with id {mission_id}")
            mission.start()
            if mission.enable_logfile:
                mission.log_file.write(
                    "{:.3f} {} SEARCH STARTED\n".format(
                        time.time() - mission.start_time, mission.host_name
                    )
                )

            reply = b"SUCCESS"
        except Exception as e:
            reply = (f"ERROR: {e}").encode()
        return reply

    def _a2s_stop_mission(self):
        """Function to stop mission

        Returns:
            bytes: SUCESS or ERROR message
        """
        try:
            mission = self._manager.get_mission()

            if mission is None:
                return b"ERROR: Mission does not exist"

            mission_id = mission.mission_id.value
            logger.info(f"Stopping mission with id {mission_id}")
            if mission.enable_logfile:
                mission.log_file.write(
                    "{:.3f} {} SEARCH STOPPED\n".format(
                        time.time() - mission.start_time, mission.host_name
                    )
                )
            mission.stop()
            self._manager.remove_mission()
            reply = b"SUCCESS"
        except Exception as e:
            reply = (f"ERROR: {e}").encode()
        finally:
            # Stop fireqos
            bandwidth_cmd = ["fireqos", "stop"]
            b = subprocess.Popen(bandwidth_cmd)
            b.communicate()
            torch.cuda.empty_cache()
            gc.collect()
        return reply

    def _a2s_get_mission_stats(self):
        """Function to send mission stats to home

        Returns:
            str: serialized MissionStats message
        """
        try:
            mission = self._manager.get_mission()
            if not mission:
                return Empty

            time_now = time.time() - mission.start_time

            if mission.enable_logfile:
                mission.log_file.write(
                    "{:.3f} {} SEARCH STATS\n".format(
                        time.time() - mission.start_time, mission.host_name
                    )
                )

            retriever_stats = mission.retriever.get_stats()
            selector_stats = mission.selector.get_stats()
            processed_objects = (
                retriever_stats.dropped_objects + selector_stats.processed_objects
            )

            mission_stats = vars(copy.deepcopy(retriever_stats))
            mission_stats.update(vars(copy.deepcopy(selector_stats)))
            keys_to_remove = [
                "total_objects",
                "processed_objects",
                "dropped_objects",
                "passed_objects",
                "false_negatives",
            ]
            for k in list(mission_stats):
                v = mission_stats[k]
                mission_stats[k] = str(v)
                if k in keys_to_remove:
                    del mission_stats[k]

            mission_stats.update(
                {
                    "server_time": str(time_now),
                    "version": str(mission._model.version),
                    "msg": "stats",
                    "ctime": str(time.ctime()),
                    "server_positives": str(mission.positives),
                    "server_negatives": str(mission.negatives),
                }
            )

            reply = MissionStats(
                totalObjects=int(retriever_stats.total_objects),
                processedObjects=int(processed_objects),
                droppedObjects=int(
                    retriever_stats.dropped_objects + selector_stats.dropped_objects
                ),
                falseNegatives=int(
                    retriever_stats.false_negatives + selector_stats.false_negatives
                ),
                others=mission_stats,
            )

            if mission.enable_logfile:
                mission.log_file.write(
                    "{:.3f} {} SEARCH STATS\n".format(
                        time.time() - mission.start_time, mission.host_name
                    )
                )

            reply = reply.SerializeToString()
        except Exception as e:
            reply = (f"ERROR: {e}").encode()
        return reply

    def _a2s_new_model(self, request: ImportModel):
        """Function to import new model from HOME

        Args:
            request (ImportModel) ImportModel message

        Returns:
            bytes: SUCESS or ERROR message

        """
        try:
            mission = self._manager.get_mission()
            model = request.model
            path = request.path
            version = model.version
            mission.import_model(model.content, path, version)
            logger.info("[IMPORT] FINISHED Model Import")
            if mission.enable_logfile:
                mission.log_file.write(
                    "{:.3f} {} IMPORT MODEL\n".format(
                        time.time() - mission.start_time, mission.host_name
                    )
                )

            reply = b"SUCCESS"
        except Exception as e:
            reply = (f"ERROR: {e}").encode()
        return reply

    def _a2s_get_test_results(self, request: str):
        """Function to test the model on the TEST dataset

        Args:
            request (str): path to the TEST dataset on the scouts

        Returns:
            str: serialized MissionResults message
        """
        try:
            test_path = Path(request)

            if not test_path.is_file():
                raise Exception

            mission = self._manager.get_mission()
            model_dir = str(mission.model_dir)
            files = sorted(glob.glob(os.path.join(model_dir, "*.*")))
            model_paths = [
                x for x in files if x.split(".")[-1].lower() in MODEL_FORMATS
            ]
            logger.info(model_paths)

            def get_version(path, idx):
                name = path.name
                try:
                    version = int(name.split("model-")[-1].split(".")[0])
                except Exception:
                    version = idx

                return version

            results = {}
            for idx, path in enumerate(model_paths):
                path = Path(path)
                version = get_version(path, idx)
                logger.info(f"model {path} version {version}")
                # create trainer and check
                # model = mission.load_model(path, version=version)
                model = self._trainer.load_model(path, version=version)
                result = model.evaluate_model(test_path)
                results[version] = result

            reply = MissionResults(results=results)
            reply = reply.SerializeToString()
        except Exception as e:
            reply = (f"ERROR: {e}").encode()
        return reply

    def _a2s_get_post_mission_archive(self):
        """Function to send mission models and logs archive

        Returns:
            bytes: mission archive zip file as a byte array
        """
        try:
            mission = self._manager.get_mission()
            data_dir = mission.data_dir
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

            mission_archive.seek(0)
            reply = mission_archive
        except Exception as e:
            reply = Empty
        return reply

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
        selector: SelectiveConfig,
        reexamination_strategy: ReexaminationStrategyConfig,
    ) -> Selector:
        if selector.HasField("topk"):
            top_k_param = json_format.MessageToDict(selector.topk)
            logger.info(f"TopK Params {top_k_param}")
            return TopKSelector(
                selector.topk.k,
                selector.topk.batchSize,
                self._get_reexamination_strategy(reexamination_strategy),
            )
        elif selector.HasField("token"):
            return TokenSelector(
                selector.token.initial_samples,
                selector.token.batch_size,
                self._get_reexamination_strategy(reexamination_strategy),
            )
        elif selector.HasField("threshold"):
            return ThresholdSelector(
                selector.threshold.threshold,
                self._get_reexamination_strategy(reexamination_strategy),
            )
        elif selector.HasField("diversity"):
            top_k_param = json_format.MessageToDict(selector.diversity)
            logger.info(f"TopK Params {top_k_param}")
            return DiversitySelector(
                selector.topk.k,
                selector.topk.batchSize,
                self._get_reexamination_strategy(reexamination_strategy),
            )
        else:
            raise NotImplementedError(
                f"unknown selector: {json_format.MessageToJson(selector)}"
            )

    def _get_reexamination_strategy(
        self, reexamination_strategy: ReexaminationStrategyConfig
    ) -> ReexaminationStrategy:
        reexamination_type = reexamination_strategy.type
        if reexamination_type == "none":
            return NoReexaminationStrategy()
        elif reexamination_type == "top":
            return TopReexaminationStrategy(reexamination_strategy.k)
        elif reexamination_type == "full":
            return FullReexaminationStrategy()
        else:
            raise NotImplementedError(
                "unknown reexamination strategy: {}".format(
                    json_format.MessageToJson(reexamination_strategy)
                )
            )

    def _get_retriever(self, dataset: Dataset) -> Retriever:
        if dataset.HasField("tile"):
            return TileRetriever(dataset.tile)
        elif dataset.HasField("frame"):
            return FrameRetriever(dataset.frame)
        elif dataset.HasField("random"):
            return RandomRetriever(dataset.random)
        elif dataset.HasField("video"):
            return VideoRetriever(dataset.video)
        else:
            raise NotImplementedError(
                f"unknown dataset: {json_format.MessageToJson(dataset)}"
            )
