# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import torch
from logzero import logger

from ...core.model_trainer import ModelTrainer
from .config import DNNRadarTrainerConfig
from .model import DNNClassifierModelRadar

if TYPE_CHECKING:
    from pathlib import Path

    from ...context.model_trainer_context import ModelContext

torch.multiprocessing.set_sharing_strategy("file_system")


class DNNClassifierTrainerRadar(ModelTrainer):
    config_class = DNNRadarTrainerConfig
    config: DNNRadarTrainerConfig

    def __init__(self, config: DNNRadarTrainerConfig, context: ModelContext) -> None:
        super().__init__(config, context)

        self.base_model_path = self.context.model_dir / "base_model.pth"
        logger.info(f" base model path: {self.base_model_path}\n")
        logger.info(f"Model_dir {self.context.model_dir}")

        logger.info("DNN CLASSIFIER TRAINER RADAR CALLED")

    def load_model(self, path: Path, version: int) -> DNNClassifierModelRadar:
        logger.info(f"Loading from path {path}")
        return DNNClassifierModelRadar(self.config, self.context, path, version)

    def train_model(self, train_dir: Path) -> DNNClassifierModelRadar:
        new_version = self.get_new_version()
        model_savepath = self.context.model_path(new_version)

        num_classes = len(self.context.class_list)
        labels = [str(label) for label in range(num_classes)]
        logger.info(f"List of labels in trainer: {labels}")

        trainpath = self.context.model_path(new_version, template="train-{}.txt")
        train_len = self.make_train_txt(trainpath, train_dir, labels)

        if self.context.check_create_test():
            valpath = self.context.model_path(new_version, template="val-{}.txt")
            val_dir = train_dir.parent / "test"
            self.make_train_txt(valpath, val_dir, labels)
        else:
            valpath = None

        num_epochs = self.config.initial_model_epochs
        if new_version > 0:
            online_epochs = self.config.online_epochs

            if isinstance(online_epochs, list):
                for epoch, pos in online_epochs:
                    if train_len["1"] >= pos:
                        num_epochs = epoch
            else:
                num_epochs = online_epochs

        cmd = [
            sys.executable,
            "-m",
            "hawk.scout.trainer.dnn_classifier_radar.train_model",
            "--trainpath",
            str(trainpath),
            "--arch",
            self.config.arch,
            "--savepath",
            str(model_savepath),
            "--num-unfreeze",
            str(self.config.unfreeze_layers),
            "--break-epoch",
            str(num_epochs),
            "--batch-size",
            str(self.config.train_batch_size),
            "--num-classes",
            str(num_classes),
        ]
        capture_files = [trainpath, train_dir]

        if new_version > 0 and self.prev_model_path is not None:
            cmd.extend(["--resume", str(self.prev_model_path)])
            # capture_files.append(self.prev_model_path)
        else:
            cmd.extend(["--base_model_path", str(self.base_model_path)])
            logger.info("EXTENDED base model path...")

        if valpath is not None:
            cmd.extend(["--valpath", str(valpath)])
            capture_files.extend([valpath, val_dir])

        cmd_str = shlex.join(cmd)
        self.capture_trainingset(cmd_str, capture_files)

        logger.info(f"TRAIN CMD\n {cmd_str}")
        subprocess.run(cmd, check=True)

        # train completed time
        train_time = time.time() - self.context.start_time

        self.prev_model_path = model_savepath
        return DNNClassifierModelRadar(
            self.config,
            self.context,
            model_savepath,
            new_version,
            train_examples=train_len,
            train_time=train_time,
        )
