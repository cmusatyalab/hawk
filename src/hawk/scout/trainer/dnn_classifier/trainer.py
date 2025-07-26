# SPDX-FileCopyrightText: 2022-2024 Carnegie Mellon University
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
from .config import DNNTrainerConfig
from .model import DNNClassifierModel

if TYPE_CHECKING:
    from pathlib import Path

    from ...context.model_trainer_context import ModelContext

torch.multiprocessing.set_sharing_strategy("file_system")


class DNNClassifierTrainer(ModelTrainer):
    config_class = DNNTrainerConfig
    config: DNNTrainerConfig

    def __init__(self, config: DNNTrainerConfig, context: ModelContext) -> None:
        super().__init__(config, context)

        logger.info(f"Model_dir {self.context.model_dir}")

        logger.info("DNN CLASSIFIER TRAINER CALLED")

    def load_model(self, path: Path, version: int) -> DNNClassifierModel:
        logger.info(f"Loading from path {path}")
        return DNNClassifierModel(self.config, self.context, path, version)

    def train_model(self, train_dir: Path) -> DNNClassifierModel:
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

        num_epochs = self.get_num_epochs(version=new_version, positives=train_len["1"])

        cmd = [
            sys.executable,
            "-m",
            "hawk.scout.trainer.dnn_classifier.train_model",
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
        return DNNClassifierModel(
            self.config,
            self.context,
            model_savepath,
            new_version,
            train_examples=train_len,
            train_time=train_time,
        )
