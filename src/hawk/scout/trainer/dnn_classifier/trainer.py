# SPDX-FileCopyrightText: 2022-2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path

import torch
from logzero import logger

from ...context.model_trainer_context import ModelContext
from ...core.config import ModelMode
from ...core.model_trainer import ModelTrainer
from .config import DNNTrainerConfig
from .model import DNNClassifierModel

torch.multiprocessing.set_sharing_strategy("file_system")


class DNNClassifierTrainer(ModelTrainer):
    config_class = DNNTrainerConfig
    config: DNNTrainerConfig

    def __init__(self, config: DNNTrainerConfig, context: ModelContext):
        super().__init__(config, context)

        logger.info(f"Model_dir {self.context.model_dir}")

        self.train_initial_model = False

        logger.info("DNN CLASSIFIER TRAINER CALLED")

    def load_model(
        self, path: Path | None = None, content: bytes = b"", version: int = -1
    ) -> DNNClassifierModel:
        new_version = self.get_new_version()

        if path is None or not path.is_file():
            assert len(content)
            path = self.context.model_path(new_version)
            path.write_bytes(content)

        version = self.get_version()
        logger.info(f"Loading from path {path}")
        self.prev_path = path
        return DNNClassifierModel(self.config, self.context, path, version)

    def train_model(self, train_dir: Path) -> DNNClassifierModel:
        # check mode if not hawk return model
        # EXPERIMENTAL
        if self.config.mode == ModelMode.ORACLE:
            return self.load_model(self.prev_path, version=0)

        elif self.config.mode == ModelMode.NOTIONAL:
            # notional_path = self.config.notional_model_path
            notional_path = self.prev_path
            # sleep for training time
            time_sleep = self.config.notional_train_time
            time_now = time.time()
            while (time.time() - time_now) < time_sleep:
                time.sleep(1)

            return self.load_model(notional_path, version=0)

        new_version = self.get_new_version()

        model_savepath = self.context.model_path(new_version)
        trainpath = self.context.model_path(new_version, template="train-{}.txt")

        # labels = [subdir.name for subdir in self._train_dir.iterdir()]
        ## reference class manager here to determine labels
        num_classes = len(self.context.class_list)
        labels = [str(label) for label in range(num_classes)]
        logger.info(f"List of labels in trainer: {labels}")
        # labels = ["0", "1"]
        train_samples = {
            label: list(train_dir.joinpath(label).glob("*")) for label in labels
        }
        train_len = {label: len(train_samples[label]) for label in labels}
        if train_len["1"] == 0:
            logger.error(train_len)
            logger.error([str(train_dir / label / "*") for label in labels])
            raise Exception

        with open(trainpath, "w") as f:
            # get easy negatives
            for easy in list(train_dir.joinpath("-1").glob("*")):
                f.write(f"{easy} 0\n")

            for label in labels:
                for path in train_samples[label]:
                    f.write(f"{path} {label}\n")

        if self.context.check_create_test():
            valpath = self.context.model_path(new_version, template="val-{}.txt")
            val_dir = train_dir.parent / "test"
            val_samples = {
                label: list(val_dir.joinpath(label).glob("*")) for label in labels
            }
            val_len = {label: len(val_samples[label]) for label in labels}
            if val_len["1"] == 0:
                logger.error(val_len)
                logger.error([str(val_dir / label / "*") for label in labels])
                raise Exception

            with open(valpath, "w") as f:
                for label in labels:
                    for path in val_samples[label]:
                        f.write(f"{path} {label}\n")
        else:
            valpath = None

        num_epochs = self.config.initial_model_epochs
        if new_version <= 0:
            self.train_initial_model = True
        else:
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

        if self.train_initial_model:
            self.train_initial_model = False
        else:
            cmd.extend(["--resume", str(self.prev_path)])
            # capture_files.append(self.prev_path)

        if valpath is not None:
            cmd.extend(["--valpath", str(valpath)])
            capture_files.extend([valpath, val_dir])

        cmd_str = shlex.join(cmd)
        self.capture_trainingset(cmd_str, capture_files)

        logger.info(f"TRAIN CMD\n {cmd_str}")
        proc = subprocess.Popen(cmd)
        proc.communicate()

        # train completed time
        train_time = time.time() - self.context.start_time

        self.prev_path = model_savepath

        return DNNClassifierModel(
            self.config,
            self.context,
            model_savepath,
            new_version,
            train_examples=train_len,
            train_time=train_time,
        )
