# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import argparse
import multiprocessing as mp
import os
import signal
import socket
import time
from datetime import datetime
from pathlib import Path

import zmq
from logzero import logger

from ..mission_config import load_config, write_config
from ..ports import H2A_PORT
from .admin import Admin
from .hawk_typing import Labeler, LabelQueueType, LabelStats, MetaQueueType
from .inbound import InboundProcess
from .outbound import OutboundProcess
from .script_labeler import ScriptLabeler
#from .ui_labeler import UILabeler
from .utils import define_scope, get_ip


# Usage: python -m hawk.home.home_main config/config.yml
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=Path, default=Path.cwd().joinpath("configs", "config.yml")
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Setting up mission
    mission_name = config.get("mission-name", "test")

    assert (
        Path(config["train_strategy"].get("initial_model_path", "")).exists()
        or Path(config["train_strategy"].get("bootstrap_path", "")).exists()
    )

    mission_id = "_".join([mission_name, datetime.now().strftime("%Y%m%d-%H%M%S")])
    scouts = config.scouts

    if config["dataset"]["type"] == "cookie":
        logger.info("Reading Scope Cookie")
        logger.info(f"Participating scouts \n{scouts}")
        config = define_scope(config)

    bandwidth = config.get("bandwidth", "100")
    assert int(bandwidth) in [
        100,
        30,
        12,
    ], f"Fireqos script may not exist for {bandwidth}"
    config["bandwidth"] = [f'[[-1, "{bandwidth}k"]]' for _ in scouts]

    # create local directories
    mission_dir = Path(config["home-params"]["mission_dir"])
    mission_dir = mission_dir / mission_id
    logger.info(mission_dir)

    log_dir = mission_dir / "logs"
    config["home-params"]["log_dir"] = str(log_dir)
    end_file = log_dir / "end"

    image_dir = mission_dir / "images"
    meta_dir = mission_dir / "meta"
    label_dir = mission_dir / "labels"

    log_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    # Save config file to log_dir
    config_path = log_dir / "hawk.yml"
    write_config(config, config_path)

    # Setting up helpers
    scout_ips = [socket.gethostbyname(scout) for scout in scouts]

    processes = []
    stop_event = mp.Event()
    meta_q: MetaQueueType = mp.Queue()
    label_q: LabelQueueType = mp.Queue()
    labelstats = LabelStats()

    try:
        # Starting home to admin conn
        context = zmq.Context()
        h2a_socket = context.socket(zmq.REQ)
        h2a_socket.connect(f"tcp://127.0.0.1:{H2A_PORT}")

        # Setup admin
        home_ip = get_ip()

        logger.info("Starting Admin Process")
        home_admin = Admin(home_ip, mission_id)
        p = mp.Process(
            target=home_admin.receive_from_home,
            kwargs={"stop_event": stop_event, "labelstats": labelstats},
        )
        processes.append(p)
        p.start()

        # Start inbound process
        logger.info("Starting Inbound Process")
        home_inbound = InboundProcess(image_dir, meta_dir, config)
        p = mp.Process(
            target=home_inbound.receive_data,
            kwargs={"result_q": meta_q, "stop_event": stop_event},
        )
        processes.append(p)
        p.start()

        # Start labeler process
        logger.info("Starting Labeler Process")
        labeler = config.get("label-mode", "ui")
        gt_dir = config["home-params"].get("label_dir", "")
        trainer = (config["train_strategy"]["type"]).lower()
        logger.info(f"Trainer {trainer}")
        label_mode = "classify"

        
        if trainer == "dnn_classifier_radar":
            if config["train_strategy"]["args"]["pick_patches"] == "True": 
                label_mode = "detect"
        elif trainer == "yolo": label_mode = "detect"

        if labeler == "script":
            home_labeler: Labeler = ScriptLabeler(label_dir, config, gt_dir, label_mode)
        elif labeler == "ui" or labeler == "browser":
            home_labeler = UILabeler(mission_dir)
        else:
            raise NotImplementedError(f"Labeler {labeler} not implemented")

        p = mp.Process(
            target=home_labeler.start_labeling,
            kwargs={
                "input_q": meta_q,
                "result_q": label_q,
                "labelstats": labelstats,
                "stop_event": stop_event,
            },
        )
        p.start()
        processes.append(p)

        # start outbound process
        logger.info("Starting Outbound Process")
        h2c_port = config.deploy.h2c_port
        home_outbound = OutboundProcess()
        p = mp.Process(
            target=home_outbound.send_labels,
            kwargs={
                "scout_ips": scout_ips,
                "h2c_port": h2c_port,
                "result_q": label_q,
                "stop_event": stop_event,
            },
        )
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
