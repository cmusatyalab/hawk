# SPDX-FileCopyrightText: 2022-2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import base64
import io
import json
import random
import socket
import threading
import time
from collections import Counter, defaultdict
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import zmq
from google.protobuf.json_format import MessageToDict
from logzero import logger
from PIL import Image

from ..mission_config import MissionConfig, load_config
from ..plugins import validate_and_scrub_config
from ..ports import H2A_PORT
from ..proto.messages_pb2 import (
    AbsolutePolicyConfig,
    ChangeDeploymentStatus,
    Dataset,
    MissionResults,
    MissionStats,
    ModelArchive,
    PercentagePolicyConfig,
    PerScoutSCMLOptions,
    ReexaminationStrategyConfig,
    RetrainPolicyConfig,
    SampleIntervalPolicyConfig,
    ScoutConfiguration,
    SelectiveConfig,
    TokenConfig,
    TopKConfig,
    TrainConfig,
)
from ..scout.core.config import ModelHomeConfig
from .stats import (
    HAWK_LABELED_CLASSES,
    HAWK_LABELED_OBJECTS,
    HAWK_UNLABELED_RECEIVED,
    collect_histogram_bucket,
    collect_metric_samples,
    collect_summary_count,
    collect_summary_total,
)

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event

LOG_INTERVAL = 15


class Admin:
    def __init__(
        self,
        home_ip: str,
        mission_id: str,
        stop_event: Event,
        explicit_start: bool = False,
    ) -> None:
        self._home_ip = home_ip
        self._start_event = threading.Event()
        self._mission_id = mission_id
        self.stop_event = stop_event
        self.explicit_start = explicit_start
        self.scout_stubs: dict[int, zmq.Socket[bytes]] = {}
        self.log_files: dict[int, TextIO] = {}
        self.test_path = ""
        self.scml = False

    def start(self) -> Admin:
        threading.Thread(target=self.receive_from_home, daemon=True).start()
        threading.Thread(target=self.get_mission_stats, daemon=True).start()
        return self

    def receive_from_home(self) -> None:
        with suppress(KeyboardInterrupt):
            # Bind H2A Server
            context = zmq.Context()
            socket = context.socket(zmq.REP)
            socket.bind(f"tcp://127.0.0.1:{H2A_PORT}")

            while not self.stop_event.is_set():
                msg_string = socket.recv_string()
                header, body = msg_string.split()
                socket.send_string("RECEIVED")

                if header == "config":
                    config_path = Path(body)
                    config = load_config(config_path)
                    self._setup_mission(config)
                else:
                    msg = f"Unknown header {header}"
                    raise NotImplementedError(msg)

    def _setup_mission(self, config: MissionConfig) -> None:
        self._mission_name = config["mission-name"]
        self.log_dir = Path(config["home-params"]["log_dir"])
        self.end_file = self.log_dir / "end"
        self.end_time = int(config.get("end-time", 5000))

        self.scouts = config.scouts

        # scout deployment status
        # self.active_scouts =
        # self.idle_scouts =

        # homeIP
        home_ip = self._home_ip

        # trainLocation
        train_location = config["train-location"]

        # missionDirectory
        mission_directory = config["scout-params"]["mission_dir"]
        self.test_path = config["scout-params"].get("test_path", "")

        # trainStrategy
        train_config = config["train_strategy"]
        train_type = train_config["type"]

        train_config.update(train_config.get("args", {}))

        if train_type == "fsl":
            support_path = train_config["example_path"]
            image = Image.open(support_path).convert("RGB")
            with io.BytesIO() as tmpfile:
                image.save(tmpfile, format="JPEG", quality=75)
                content = tmpfile.getvalue()
            train_config["support_data"] = base64.b64encode(content).decode("utf8")

        def content_from_file(file: Path | None) -> bytes | None:
            if file is None or not file.is_file():
                return None
            return file.read_bytes()

        def modelarchive_from_file(file: Path | None) -> ModelArchive | None:
            model_content = content_from_file(file)
            if not model_content:
                return None
            return ModelArchive(content=model_content)

        # pull settings from train_config that are only used by home
        model_home_config = ModelHomeConfig.model_validate(train_config)

        bootstrap_zip = content_from_file(model_home_config.bootstrap_path) or b""
        initial_model = modelarchive_from_file(model_home_config.initial_model_path)
        base_model = modelarchive_from_file(model_home_config.base_model_path)
        train_validate = model_home_config.train_validate

        train_config = validate_and_scrub_config("model", train_type, train_config)
        train_strategy = TrainConfig(trainer=train_type, config=train_config)

        # retrainPolicy
        retrain_config = config["retrain_policy"]
        retrain_type = retrain_config["type"]
        if retrain_type == "percentage":
            retrain_policy = RetrainPolicyConfig(
                percentage=PercentagePolicyConfig(
                    threshold=retrain_config["threshold"],
                    onlyPositives=retrain_config["only_positives"],
                ),
            )
        elif retrain_type == "absolute":
            retrain_policy = RetrainPolicyConfig(
                absolute=AbsolutePolicyConfig(
                    threshold=retrain_config["threshold"],
                    onlyPositives=retrain_config["only_positives"],
                ),
            )
        elif retrain_type == "sample":
            retrain_policy = RetrainPolicyConfig(
                sample=SampleIntervalPolicyConfig(
                    num_intervals=retrain_config["num_intervals"],
                ),
            )
        else:
            errmsg = f"Unknown retrain policy {retrain_type}"
            raise NotImplementedError(errmsg)

        # dataset
        dataset_config = config["dataset"]
        dataset_type = dataset_config["type"]

        if "index_path" in dataset_config:
            logger.info(f"Index {dataset_config['index_path']}")

        self.class_list = dataset_config.get("class_list", ["positive"])
        logger.info(f"Class list: {self.class_list}")

        datasets = {}
        for index, _scout in enumerate(self.scouts):
            if dataset_type == "network":
                network_config = dataset_config["network"]
                config_dict = {
                    "index_path": dataset_config["index_path"],
                    "server_host": network_config["server_address"],
                    "server_port": network_config["server_port"],
                    "balance_mode": dataset_config.get("data_rate_balance"),
                    "tiles_per_frame": dataset_config.get("tiles_per_frame"),
                    "timeout": dataset_config.get("timeout"),
                }
            elif dataset_type == "video":
                config_dict = {
                    "video_path": dataset_config["video_list"][index],
                    "timeout": dataset_config.get("timeout"),
                }
            else:
                config_dict = dataset_config

            if dataset_type == "cookie":
                dataset_type = "random"

            dataset_conf = validate_and_scrub_config(
                "retriever",
                dataset_type,
                config_dict,
                inject={"mission_id": "", "data_root": "/"},
            )

            datasets[index] = Dataset(
                retriever=dataset_type,
                config=dataset_conf,
            )

        # reexamination
        reexamination_config = config["reexamination"]
        reexamination_type = reexamination_config.get("type", "top")

        reexamination = None
        if reexamination_type == "top":
            k_value = reexamination_config.get("k", 100)
            reexamination = ReexaminationStrategyConfig(
                type=reexamination_type,
                k=k_value,
            )
        elif reexamination_type in {"full", "none"}:
            reexamination = ReexaminationStrategyConfig(
                type=reexamination_type,
            )
        else:
            errmsg = f"Unknown reexamination {reexamination_type}"
            raise NotImplementedError(errmsg)

        # selector
        selector_config = config["selector"]
        selector_type = selector_config.get("type", "topk")

        if selector_type == "topk":
            topk_config = selector_config.get("topk", {})
            k_value = topk_config.get("k", 10)
            batch_size = topk_config.get("batchSize", 1000)
            countermeasure_threshold = topk_config.get("countermeasure_threshold", 0.5)
            total_countermeasures = topk_config.get("total_countermeasures", 700)
            selector = SelectiveConfig(
                topk=TopKConfig(
                    k=k_value,
                    batchSize=batch_size,
                    countermeasure_threshold=countermeasure_threshold,
                    total_countermeasures=total_countermeasures,
                ),
            )
        elif selector_type == "token":
            token_config = selector_config.get("token", {})
            init_samples = token_config.get("initial_samples")
            batch_size = token_config.get("batch_size")
            countermeasure_threshold = token_config.get("countermeasure_threshold", 0.5)
            total_countermeasures = token_config.get("total_countermeasures", 700)
            upper_thresh_start = token_config.get("upper_threshold_start", 1.0)
            upper_thresh_delta = token_config.get("upper_threshold_delta", 1.0)
            lower_thresh_start = token_config.get("lower_threshold_start", 1.0)
            lower_thresh_delta = token_config.get("lower_threshold_delta", 1.0)
            sliding_window = token_config.get("sliding_window", False)
            selector = SelectiveConfig(
                token=TokenConfig(
                    initial_samples=init_samples,
                    batch_size=batch_size,
                    countermeasure_threshold=countermeasure_threshold,
                    total_countermeasures=total_countermeasures,
                    upper_threshold_start=upper_thresh_start,
                    upper_threshold_delta=upper_thresh_delta,
                    lower_threshold_start=lower_thresh_start,
                    lower_threshold_delta=lower_thresh_delta,
                    sliding_window=sliding_window,
                ),
            )
        else:
            errmsg = f"Unknown selector {selector_type}"
            raise NotImplementedError(errmsg)

        # SCML deployment options
        deployment_options = config.get("scml_deploy_options", "")
        self.scout_deployment_status = {}  ## Active, Idle, or Dead
        scml_deploy_options = {
            scout: PerScoutSCMLOptions(scout_dict={}) for scout in self.scouts
        }

        if deployment_options:
            self.scml = True
            default_deploy_scout = deployment_options.get("default_deploy_scout", [])
            if default_deploy_scout:
                for scout in self.scouts:
                    for deploy_option in default_deploy_scout:
                        scml_deploy_options[scout].scout_dict[deploy_option] = (
                            default_deploy_scout[deploy_option]
                        )  ## populate default values for present fields

            ## home conditions to trigger new scout activation
            default_deploy_home = deployment_options["default_deploy_home"]
            self.min_num_scout_destroyed = default_deploy_home.get(
                "min_num_scouts_destroyed",
                0,
            )
            self.min_num_scouts_remaining = default_deploy_home.get(
                "min_num_scouts_remaining",
                0,
            )
            self.deploy_on_any_loss = default_deploy_home.get(
                "deploy_on_any_loss",
                False,
            )

            ## Add the per scout override to set conditions for individual scouts.
            per_scout_override = deployment_options.get("per_scout_override", [])
            if per_scout_override:
                for scout in per_scout_override:
                    scout_options = per_scout_override[scout]
                    if scout_options:  ## if actually has fields to override
                        for option in scout_options:
                            # if set, updates any scout unique configurations
                            # from the default values
                            scml_deploy_options[scout].scout_dict[option] = (
                                scout_options[option]
                            )

            for scout in self.scouts:
                if (
                    scml_deploy_options[scout].scout_dict.get("start_time", 0) > 0
                    or scml_deploy_options[scout].scout_dict.get("start_on_model", 0)
                    > 0
                    or scml_deploy_options[scout].scout_dict.get(
                        "start_mission_duration_percentage",
                        0,
                    )
                    > 0
                ):
                    self.scout_deployment_status[scout] = "Idle"
                else:
                    self.scout_deployment_status[scout] = "Active"
        else:
            (
                default_start_time,
                default_on_model,
                default_mission_duration_percentage,
            ) = (0, 0, 0)

            scml_deploy_options = {}
            for scout in self.scouts:  ## for no scml basic config. LL as normal
                scml_deploy_options[scout] = PerScoutSCMLOptions(
                    scout_dict={
                        "start_time": default_start_time,
                        "start_on_model": default_on_model,
                        "mission_percentage": default_mission_duration_percentage,
                    },
                )
            if (
                default_start_time > 0
                or default_on_model > 0
                or default_mission_duration_percentage > 0
            ):
                self.scout_deployment_status[scout] = "Idle"
            else:
                self.scout_deployment_status[scout] = "Active"

        ## subclass and novel class discovery
        self.novel_class_discovery = config.get("novel_class_discovery", False)
        self.sub_class_discovery = config.get("sub_class_discovery", False)

        # bandwidthFunc
        bw_config = config["bandwidth"]
        logger.info(f"BW config: {bw_config}")
        assert len(self.scouts) == len(bw_config), (
            f"Length Bandwidth {len(bw_config)} does not match {len(self.scouts)}"
        )
        bandwidth_func = {}
        for i, _b in enumerate(bw_config):
            bandwidth_func[int(i)] = str(_b)

        self._zmq_context = zmq.Context()
        a2s_port = config.deploy.a2s_port

        self.scout_stubs = {}
        self.stub_socket = {}
        self.socket_stub = {}
        for index, host in enumerate(self.scouts):
            ip = socket.gethostbyname(host)
            stub = self._zmq_context.socket(zmq.REQ)
            stub.connect(f"tcp://{ip}:{a2s_port}")
            self.scout_stubs[index] = stub
            self.stub_socket[host] = stub
            self.socket_stub[stub] = host

        # setup ScoutConfiguration
        # Call a2s_configure_scout and wait for success message

        # inform scouts on which address/port they can find each other
        s2s_port = config.deploy.s2s_port
        s2s_scouts = [f"{host}:{s2s_port}" for host in self.scouts]

        logger.info(self._mission_id)
        logger.info(s2s_scouts)

        for index, stub in self.scout_stubs.items():
            scout_config = ScoutConfiguration(
                missionId=self._mission_id,
                scouts=s2s_scouts,
                scoutIndex=index,
                homeIP=home_ip,
                trainLocation=train_location,
                missionDirectory=mission_directory,
                trainStrategy=train_strategy,
                retrainPolicy=retrain_policy,
                dataset=datasets[index],
                selector=selector,
                reexamination=reexamination,
                initialModel=initial_model,
                baseModel=base_model,
                bootstrapZip=bootstrap_zip,
                bandwidthFunc=bandwidth_func,
                validate=train_validate,
                class_list=self.class_list,
                scml_deploy_opts=scml_deploy_options[self.scouts[index]],
                novel_class_discovery=self.novel_class_discovery,
                sub_class_discovery=self.sub_class_discovery,
            )
            msg = [
                b"a2s_configure_scout",
                scout_config.SerializeToString(),
            ]
            stub.send_multipart(msg)

        return_msgs = {}
        for index, stub in self.scout_stubs.items():
            reply = stub.recv()
            return_msgs[index] = reply.decode()

        # Remove scouts that failed, create a list to make sure we can safely delete
        known_scouts = list(self.scout_stubs.keys())
        for index in known_scouts:
            if index not in return_msgs or "ERROR" in return_msgs[index]:
                errormsg = return_msgs.get(index, "No response")
                scout = self.scouts[index]
                logger.error(
                    f"ERROR during Configuration from Scout {scout}: {errormsg}",
                )
                del self.scout_stubs[index]

        if not self.explicit_start:
            self.start_mission()

    def start_mission(self) -> None:
        """Explicit start Mission command."""
        # Start Mission

        logger.info("Starting mission")
        self.log_files = {
            index: self.log_dir.joinpath(f"get-stats-{index}.txt").open("a")
            for index in self.scout_stubs
        }

        self.start_time = time.time()
        for stub in self.scout_stubs.values():
            msg = [
                b"a2s_start_mission",
                self._mission_id.encode("utf-8"),
            ]
            stub.send_multipart(msg)

        for stub in self.scout_stubs.values():
            stub.recv()

        logger.info("Start msg received")

        if self.scml:
            threading.Thread(target=self.scml_deployment_status, daemon=True).start()
        ## start tracking scml deployment status of each scout

    def stop_mission(self) -> None:
        """Explicit stop Mission command."""
        # close per-scout log files to suspend stats collection
        log_files, self.log_files = self.log_files, {}
        for log_file in log_files.values():
            log_file.close()

        for stub in self.scout_stubs.values():
            msg = [
                b"a2s_stop_mission",
                self._mission_id.encode("utf-8"),
            ]
            stub.send_multipart(msg)

        for stub in self.scout_stubs.values():
            logger.info(f"Stub: {stub}")
            stub.recv()

    def scml_deployment_status(self) -> None:
        ## loop to get deployment status from each scout, and check if scout has
        ## died, deploy new scout if necessary.
        dead_scouts = []
        while not self.stop_event.is_set():
            new_dead_scouts = 0
            ## get the current deployment statuses from all scouts
            for stub in self.scout_stubs.values():
                msg = [b"a2s_sync_deploy_status"]
                stub.send_multipart(msg)
            for stub in self.scout_stubs.values():
                response = stub.recv()
                self.scout_deployment_status[self.socket_stub[stub]] = (
                    response.decode()
                )  ## set the current scout status
                ## check if scout went from active to idle (died or some other reason)
                if (
                    self.scout_deployment_status[self.socket_stub[stub]] == "Dead"
                    and self.socket_stub[stub] not in dead_scouts
                ):
                    dead_scouts.append(self.socket_stub[stub])
                    new_dead_scouts += 1
                # logger.info(
                #     f"Scout Status: {self.socket_stub[stub]}, "
                #     f"{self.scout_deployment_status[self.socket_stub[stub]]}"
                # )
            active_scouts = [
                scout
                for scout in self.scout_deployment_status
                if self.scout_deployment_status[scout] == "Active"
            ]
            idle_scouts = [
                scout
                for scout in self.scout_deployment_status
                if self.scout_deployment_status[scout] == "Idle"
            ]
            logger.info(f"Active scouts: {active_scouts}")
            logger.info(f"Idle scouts: {idle_scouts}")

            ## Check whether scout has died and how many idle scouts to deploy,
            ## according to configurations
            if new_dead_scouts > 0:
                if self.deploy_on_any_loss:  ## from home deploy config
                    ## deploy 1 new scout for each that has died (if available)
                    num_scouts_to_deploy = min(new_dead_scouts, len(idle_scouts))
                elif len(active_scouts) < self.min_num_scouts_remaining:
                    ## deploy number of scouts up to threshold number (if available)
                    num_scouts_to_deploy = min(
                        self.min_num_scouts_remaining - len(active_scouts),
                        len(idle_scouts),
                    )
                for _ in range(num_scouts_to_deploy):
                    activating_scout = idle_scouts.pop(
                        random.randint(0, len(idle_scouts) - 1),
                    )  ## pick a random idle scout to deploy
                    data = ChangeDeploymentStatus(ActiveStatus=True)
                    msg = [b"a2s_change_deploy_status", data.SerializeToString()]
                    self.stub_socket[activating_scout].send_multipart(msg)
                    self.stub_socket[activating_scout].recv()

            time.sleep(10)

    def get_mission_stats(self) -> None:
        count = 1
        finish_time = 1e14
        processed_complete = False
        # prev_bytes = prev_processed = 0
        try:
            while not self.stop_event.is_set():
                time.sleep(LOG_INTERVAL)

                # wait until mission has started
                if not self.log_files:
                    continue

                stats = self.accumulate_mission_stats()

                negatives = collect_histogram_bucket(HAWK_LABELED_OBJECTS, 0)
                positives = collect_summary_count(HAWK_LABELED_OBJECTS) - negatives

                per_class_counts: Counter[str] = Counter()
                for sample in collect_metric_samples(HAWK_LABELED_CLASSES):
                    if (class_name := sample.labels["class_name"]) != "negative":
                        per_class_counts[class_name] += int(sample.value)

                stats.update(
                    {
                        "positives": positives,
                        "negatives": negatives,
                        "bytes": collect_summary_total(HAWK_UNLABELED_RECEIVED),
                        "count_by_class": dict(per_class_counts),
                    },
                )

                log_path = self.log_dir / f"stats-{count:06}.json"
                with open(log_path, "w") as f:
                    stats["home_time"] = time.time() - self.start_time
                    json.dump(stats, f, indent=4, sort_keys=True)

                # update a symlink pointing at the new stats-00000N.json file
                # for the streamlit gui, using a temporary link and rename to
                # update the link atomically.
                mission_stats = self.log_dir / "mission-stats.json"
                tmplink = mission_stats.with_suffix(".tmp")
                tmplink.unlink(missing_ok=True)
                tmplink.symlink_to(log_path.name, target_is_directory=False)
                tmplink.replace(mission_stats)

                if stats["home_time"] > self.end_time:
                    logger.info("End mission")
                    with open(self.end_file, "w") as f:
                        f.write("\n")
                    break

                if (
                    stats["processedObjects"] != 0
                    and stats["processedObjects"] == stats["totalObjects"]
                ):
                    logger.info("Processed all objects, waiting 60 seconds...")
                    if not processed_complete:
                        finish_time = time.time() + 60
                        processed_complete = True
                    # time.sleep(60)
                    # self.stop_event.set()
                    # logger.info("End mission")
                    # with open(self.end_file, "w") as f:
                    #    f.write("\n")
                    # break

                # prev_bytes = stats["bytes"]
                # prev_processed = stats["processedObjects"]
                count += 1
                if finish_time < time.time():
                    break
        except (Exception, KeyboardInterrupt):
            logger.error("Exception in get_mission_stats")
            raise
        finally:
            self.stop_event.set()

    def get_post_mission_archive(self) -> None:
        for index, stub in self.scout_stubs.items():
            msg = [
                b"a2s_get_post_mission_archive",
                b"",
            ]
            stub.send_multipart(msg)
            reply = stub.recv()

            if len(reply):
                with open(f"mission_{index}.zip", "wb") as f:
                    f.write(reply)

    def get_test_results(self) -> None:
        assert len(self.test_path), "Test path not provided"
        for stub in self.scout_stubs.values():
            msg = [
                b"a2s_get_test_results",
                self.test_path.encode("utf-8"),
            ]
            stub.send_multipart(msg)

        for index, stub in self.scout_stubs.items():
            reply = stub.recv()
            hostname = self.scouts[index].split(".")[0]
            results_dir = Path(self.log_dir.parent) / "results" / f"{hostname}"
            results_dir.mkdir(parents=True, exist_ok=True)
            if len(reply):
                try:
                    mission_stat_msg = MissionResults()
                    mission_stat_msg.ParseFromString(reply)
                    mission_stat = mission_stat_msg.results

                    for version, result in mission_stat.items():
                        model_stat = MessageToDict(result)
                        stat_path = results_dir / f"model-result-{version:06}.json"
                        with open(stat_path, "w") as f:
                            json.dump(model_stat, f, indent=4, sort_keys=True)
                except Exception:
                    errmsg = reply.decode()
                    logger.error(f"ERROR during Testing from Scout {index}\n {errmsg}")

    def accumulate_mission_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = defaultdict(int)
        str_ignore = [
            "server_time",
            "ctime",
            "train_positives",
            "server_positives",
            "msg",
        ]
        str_foreach = [
            "mission_state",
        ]
        single = ["server_time", "train_positives", "version"]
        for stub in self.scout_stubs.values():
            msg = [
                b"a2s_get_mission_stats",
                self._mission_id.encode("utf-8"),
            ]
            stub.send_multipart(msg)

        for index, stub in self.scout_stubs.items():
            reply = stub.recv()
            mission_stat_msg = MissionStats()
            mission_stat_msg.ParseFromString(reply)
            mission_stat = MessageToDict(mission_stat_msg)
            self.log_files[index].write(json.dumps(mission_stat) + "\n")

            for k, v in mission_stat.items():
                if isinstance(v, dict):
                    others = v
                    for key, value in others.items():
                        if key in mission_stat:
                            continue
                        if key in str_foreach:
                            if index == 0:
                                stats[key] = []
                            stats[key].append(value)
                        elif key in str_ignore:
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
