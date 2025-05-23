# SPDX-FileCopyrightText: 2022-2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import argparse
import atexit
import multiprocessing as mp
import os
import sys
import time
from contextlib import suppress
from datetime import datetime
from pathlib import Path

import logzero
import zmq
from logzero import logger
from prometheus_client import start_http_server as start_metrics_server

from ..classes import ClassList
from ..mission_config import load_config, write_config
from ..ports import H2A_PORT, HOME_METRICS_PORT
from .admin import Admin
from .home_labeler import LabelerDiskQueue
from .home_scout import ScoutQueue, Strategy
from .script_labeler import ScriptLabeler
from .stats import HAWK_MISSION_STATUS
from .utils import define_scope, get_ip


def daemonize(mission_dir: Path) -> None:
    # fork-detach-fork
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
        os.setsid()
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        sys.stderr.write(f"Failed to daemonize process {e.strerror}")
        sys.exit(1)

    logzero.json()

    # redirect stdio
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_RDONLY)
    logfile = os.open(
        mission_dir / "hawk_home.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND
    )
    os.dup2(devnull, sys.stdin.fileno())
    os.dup2(logfile, sys.stdout.fileno())
    os.dup2(logfile, sys.stderr.fileno())


def resolve_path(config: dict[str, str], key: str, mission_dir: Path) -> None:
    with suppress(KeyError):
        local_path = Path(config[key])
        if not local_path.is_absolute():
            local_path = mission_dir.joinpath(local_path).resolve()
            config[key] = str(local_path)


# Usage: python -m hawk.home.home_main config/config.yml
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--detach", action="store_true", help="Run in the detached mode"
    )
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
    mission_status = HAWK_MISSION_STATUS.labels(mission=mission_id)
    mission_status.state("starting")

    if config["dataset"]["type"] == "cookie":
        logger.info("Reading Scope Cookie")
        logger.info(f"Participating scouts \n{config.scouts}")
        config = define_scope(config)

    bandwidth = config.get("bandwidth", "100")
    assert int(bandwidth) in [
        100,
        50,
        30,
        12,
    ], f"Fireqos script may not exist for {bandwidth}"
    config["bandwidth"] = [f'[[-1, "{bandwidth}k"]]' for _ in config.scouts]

    # create new mission directory if we've been started with a config.yml
    if mission_dir is None:
        mission_dir = Path(config["home-params"]["mission_dir"], mission_id)
        mission_dir.mkdir(parents=True)

    if args.detach:
        daemonize(mission_dir)

    # write pidfile
    pidfile = mission_dir / "hawk_home.pid"
    atexit.register(os.remove, str(pidfile))
    my_pid = str(os.getpid())
    pidfile.write_text(f"{my_pid}\n")

    logger.info(mission_dir)

    # create local directories
    log_dir = mission_dir / "logs"
    trace_dir = mission_dir / "traces"
    config["home-params"]["log_dir"] = str(log_dir)
    end_file = log_dir / "end"

    log_dir.mkdir(parents=True)
    trace_dir.mkdir()

    # resolve local paths relative to mission_dir
    resolve_path(config["train_strategy"], "bootstrap_path", mission_dir)
    resolve_path(config["train_strategy"], "initial_model_path", mission_dir)

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
            start_metrics_server(port=metrics_port)
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

        class_names = config.get("dataset", {}).get("class_list", ["positive"])
        class_list = ClassList(class_names)

        # Start scout and labeler queues
        queues = config.get("queue-mode", "thread")
        if queues == "thread":
            logger.info("Initializing Scout Queue")
            strategy = config.get("label-queue-strategy", Strategy.FIFO)
            scout_queue = ScoutQueue(
                mission_id=mission_id,
                mission_dir=mission_dir,
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
                class_list=class_list,
                label_queue_size=label_queue_max,
            ).start()

            logger.info("Starting Scout Queue")
            scout_queue.start()

        # Start sub class clustering process if set in config file
        if config.get("sub_class_discovery", False):
            from . import sub_class_clustering

            p = mp.Process(target=sub_class_clustering.main, args=(mission_dir,))
            processes.append(p)
            p.start()

        # Send config file to admin
        # send msg "<config> <path to config file>"

        mission_status.state("configuring")
        h2a_socket.send_string(f"config {config_path}")
        h2a_socket.recv()
        mission_status.state("running")

        while not stop_event.is_set():
            if end_file.is_file():
                stop_event.set()
            time.sleep(10)

        logger.info("Stop event is set")
    except KeyboardInterrupt as e:
        logger.error(e)
    finally:
        mission_status.state("stopped")
        time.sleep(10)
        home_admin.stop_mission()
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
