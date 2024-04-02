# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import sys
from contextlib import suppress
from pathlib import Path

from ..mission_config import MissionConfig
from .to_labeler import LabelerDiskQueue
from .to_scout import ScoutQueue, Strategy


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator", type=int)
    parser.add_argument("--label-queue-size", type=int)
    parser.add_argument(
        "--label-queue-strategy",
        type=Strategy,
        choices=list(Strategy),
    )
    parser.add_argument("mission_directory", type=Path, nargs="?", default=".")

    args = parser.parse_args()

    config_file = args.mission_directory / "mission_config.yml"
    config = MissionConfig.from_yaml(config_file.read_text())

    if args.label_queue_size is None:
        args.label_queue_size = config.get("label-queue-max", 0)

    if args.label_queue_strategy is None:
        args.label_queue_strategy = config.get("label-queue-strategy", Strategy.FIFO)

    with suppress(KeyboardInterrupt):
        scout_queue = ScoutQueue(
            strategy=args.label_queue_strategy,
            scouts=config.scouts,
            h2c_port=config.deploy.h2c_port,
            coordinator=args.coordinator,
        )

        LabelerDiskQueue(
            scout_queue=scout_queue,
            mission_dir=args.mission_directory,
            label_queue_size=args.label_queue_size,
        ).start()

        scout_queue.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
