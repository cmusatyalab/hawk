# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import argparse
import multiprocessing as mp
import socket
import time
from datetime import datetime
from pathlib import Path

import zmq
from logzero import logger

from ..mission_config import load_config, write_config
from ..ports import H2A_PORT
from .admin import Admin
from .inbound import InboundProcess
from .outbound import OutboundProcess
from .script_labeler import ScriptLabeler
from .stats import LabelStats
from .utils import define_scope, get_ip


# Usage: python -m hawk.home.home_main config/config.yml
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=Path, default=Path.cwd().joinpath("configs", "config.yml")
    )
    args = parser.parse_args()

    # handle running from existing mission directory
    if args.config.is_dir():
        mission_dir = args.config
        args.config = mission_dir / "mission_config.yml"
    else:
        mission_dir = None

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

    # create new mission directory if we've been started with a config.yml
    if mission_dir is None:
        mission_dir = Path(config["home-params"]["mission_dir"])
        mission_dir = mission_dir / mission_id
    logger.info(mission_dir)

    # create local directories
    log_dir = mission_dir / "logs"
    config["home-params"]["log_dir"] = str(log_dir)
    end_file = log_dir / "end"

    log_dir.mkdir(parents=True)

    results_jsonl = mission_dir / "unlabeled.jsonl"
    labeled_jsonl = mission_dir / "labeled.jsonl"

    # Save final config file to log_dir
    # this includes scope derived parameters, normalized bandwidth, and
    # home-params.log_dir
    # TODO some of this could move into initializers of a config dataclass
    config_path = log_dir / "hawk.yml"
    write_config(config, config_path)

    # Setting up helpers
    scout_ips = [socket.gethostbyname(scout) for scout in scouts]

    processes = []
    stop_event = mp.Event()
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

        # Start labeler process
        labeler = config.get("label-mode", "ui")
        if labeler == "script":
            logger.info("Starting Labeler Process")
            labeler = ScriptLabeler.from_mission_config(config, mission_dir)

            p = mp.Process(target=labeler.run)
            p.start()
            processes.append(p)

        # Start inbound process
        logger.info("Starting Inbound Process")
        strategy = config.get("label-queue-strategy", "fifo")
        label_queue_max = config.get("label-queue-max", 0)
        label_sem = mp.BoundedSemaphore(label_queue_max) if label_queue_max else None
        home_inbound = InboundProcess(
            results_jsonl, strategy, len(config.deploy.scouts)
        )
        p = mp.Process(
            target=home_inbound.scout_to_labeler,
            kwargs={"next_label": label_sem, "stop_event": stop_event},
        )
        p.start()
        processes.append(p)

        # start outbound process
        logger.info("Starting Outbound Process")
        home_outbound = OutboundProcess(
            scout_ips, config.deploy.h2c_port, labeled_jsonl
        )
        p = mp.Process(
            target=home_outbound.labeler_to_scout,
            kwargs={
                "next_label": label_sem,
                "stop_event": stop_event,
                "labelstats": labelstats,
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
    except KeyboardInterrupt:
        logger.info("Stopping mission")
    except BaseException as e:
        logger.error(e)
    finally:
        stop_event.set()
        home_admin.stop_mission()
        time.sleep(10)
        for p in processes:
            p.terminate()
        logger.info("Mission stopped")

        # pid = os.getpid()
        # os.kill(pid, signal.SIGKILL)

        # Uncomment to delete mission dir after mission

        # if mission_dir.is_dir():
        #     logger.info("Deleting directory {}".format(mission_dir))
        #     shutil.rmtree(mission_dir)


if __name__ == "__main__":
    main()
