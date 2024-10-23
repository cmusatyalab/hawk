# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import time

from logzero import logger


def clustering_function(file_name, novel_class_discovery, sub_class_discovery) -> None:
    while True:
        logger.info("Running novel class discovery...")
        time.sleep(5)


def send_samples() -> None:
    pass
