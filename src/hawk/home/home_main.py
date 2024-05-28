# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import argparse
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path

import zmq
from logzero import logger
from prometheus_client import start_http_server as start_metrics_server

from ..mission_config import load_config, write_config
from ..ports import H2A_PORT, HOME_METRICS_PORT
from .admin import Admin
from .script_labeler import ScriptLabeler
from .to_labeler import LabelerDiskQueue
from .to_scout import ScoutQueue, Strategy
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

    if config["dataset"]["type"] == "cookie":
        logger.info("Reading Scope Cookie")
        logger.info(f"Participating scouts \n{config.scouts}")
        config = define_scope(config)

    bandwidth = config.get("bandwidth", "100")
    assert int(bandwidth) in [
        100,
        30,
        12,
    ], f"Fireqos script may not exist for {bandwidth}"
    config["bandwidth"] = [f'[[-1, "{bandwidth}k"]]' for _ in config.scouts]

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

    # Save final config file to log_dir
    # this includes scope derived parameters, normalized bandwidth, and
    # home-params.log_dir
    # TODO some of this could move into initializers of a config dataclass
    config_path = log_dir / "hawk.yml"
    write_config(config, config_path)

    processes = []
    stop_event = mp.Event()

    try:
        # Starting home to admin conn
        context = zmq.Context()
        h2a_socket = context.socket(zmq.REQ)
        h2a_socket.connect(f"tcp://127.0.0.1:{H2A_PORT}")

        # Setup admin
        home_ip = get_ip()

        logger.info("Starting Admin Thread")
        home_admin = Admin(home_ip, mission_id, stop_event)
        home_admin.start()

        try:
            metrics_port = int(
                config.get("home-params", {}).get("metrics-port", HOME_METRICS_PORT)
            )
            start_metrics_server(port=metrics_port, addr="127.0.0.1")
        except ValueError:
            pass

        # Start labeler process
        labeler = config.get("label-mode", "ui")
        trainer = (config["train_strategy"]["type"]).lower()
        logger.info(f"Trainer {trainer}")

        if labeler == "script":
            logger.info("Starting Labeler Process")
            labeler = ScriptLabeler.from_mission_config(config, mission_dir)

            p = mp.Process(target=labeler.run)
            processes.append(p)
            p.start()

        # Start scout and labeler queues
        queues = config.get("queue-mode", "thread")
        if queues == "thread":
            logger.info("Initializing Scout Queue")
            strategy = config.get("label-queue-strategy", Strategy.FIFO)
            scout_queue = ScoutQueue(
                mission_id=mission_id,
                strategy=strategy,
                scouts=config.scouts,
                h2c_port=config.deploy.h2c_port,
                zmq_context=context,
            )

            logger.info("Starting Labeler Queue")
            label_queue_max = config.get("label-queue-max", 0)
            LabelerDiskQueue(
                mission_id=mission_id,
                scout_queue=scout_queue,
                mission_dir=mission_dir,
                label_queue_size=label_queue_max,
            ).start()

            logger.info("Starting Scout Queue")
            scout_queue.start()

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
        # home_admin.stop_mission()
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
