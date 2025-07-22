# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import glob
import shlex
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from logzero import logger

from ...context.model_trainer_context import ModelContext
from ...core.model_trainer import ModelTrainer
from ...core.utils import log_exceptions
from .config import YOLOTrainerConfig
from .model import YOLOModel

torch.multiprocessing.set_sharing_strategy("file_system")


class YOLOTrainer(ModelTrainer):
    config_class = YOLOTrainerConfig
    config: YOLOTrainerConfig

    def __init__(self, config: YOLOTrainerConfig, context: ModelContext):
        super().__init__(config, context)

        logger.info(f"Model_dir {self.context.model_dir}")

        if self.config.test_dir is not None:
            msg = f"Test Path {self.config.test_dir} provided does not exist"
            assert self.config.test_dir.exists(), msg

        logger.info("YOLO TRAINER CALLED")

    def load_model(self, path: Path, version: int) -> YOLOModel:
        self.context.stop_model()
        logger.info(f" Trainer Loading from path {path}")
        return YOLOModel(self.config, self.context, path, version)

    @log_exceptions
    def train_model(self, train_dir: Path) -> YOLOModel:
        new_version = self.get_new_version()

        model_savepath = self.context.model_path(new_version, template="model-{}.pt")
        trainpath = self.context.model_path(new_version, template="train-{}.txt")

        # can change this to also train on empty (no detection) images in 0/
        # directory from examples dir
        labels = ["1"]
        train_samples = {
            label: glob.glob(str(train_dir / label / "*")) for label in labels
        }
        train_len = {label: len(train_samples[label]) for label in labels}
        if train_len["1"] == 0:
            logger.error(train_len)
            logger.error([str(train_dir / label / "*") for label in labels])
            raise Exception

        with open(trainpath, "w") as f:
            for label in labels:
                for sample in train_samples[label]:
                    f.write(f"{sample}\n")

        noval = True
        if self.config.test_dir is not None:
            noval = False
            valpath = self.context.model_path(new_version, template="val-{}.txt")
            with open(valpath, "w") as f:
                for path in self.config.test_dir.glob("*/*"):
                    f.write(f"{path}\n")

        num_epochs = self.config.initial_model_epochs
        if new_version > 0:
            online_epochs = self.config.online_epochs

            if isinstance(online_epochs, list):
                for epoch, pos in online_epochs:
                    if train_len["1"] >= pos:
                        num_epochs = epoch
            else:
                num_epochs = online_epochs

        ## NEED TO MODIFY "nc" and names according to the class list.
        data_dict = {
            "path": str(self.context.model_dir),
            "train": str(trainpath),
            "nc": 1,
            "names": ["positive"],
        }

        if not noval:
            data_dict["val"] = valpath

        data_file = self.context.model_dir / "data.yaml"
        with open(data_file, "w") as outfile:
            yaml.dump(data_dict, outfile, default_flow_style=False)

        if new_version <= 0 or self.prev_model_path is None:
            weights = Path("yolov5s.pt")
        else:
            weights = self.prev_model_path

        cmd = [
            sys.executable,
            "-m",
            "hawk.scout.trainer.yolo.yolov5.train",
            "--savepath",
            str(model_savepath),
            "--epochs",
            str(num_epochs),
            "--batch-size",
            str(self.config.train_batch_size),
            "--weights",
            str(weights),
            "--imgsz",
            str(self.config.image_size),
            "--data",
            str(data_file),
            # "--freeze", ## if wanting to freeze backbone
            # str(10),
        ]
        capture_files = [data_file, trainpath, train_dir]

        # if new_version > 0:
        #     capture_files.append(weights)

        if self.config.test_dir is not None:
            cmd.extend(["--noval", "False"])
            capture_files.extend([valpath, self.config.test_dir])

        cmd_str = shlex.join(cmd)
        self.capture_trainingset(cmd_str, capture_files)

        logger.info(f"TRAIN CMD \n {cmd_str}")
        proc = subprocess.Popen(cmd)
        proc.communicate()
        if not model_savepath.exists():
            raise FileNotFoundError
        logger.info("Training completed")

        self.context.stop_model()

        self.prev_model_path = model_savepath
        return YOLOModel(
            self.config,
            self.context,
            model_savepath,
            new_version,
            train_examples=train_len,
        )
