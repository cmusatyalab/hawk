# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml
from logzero import logger

from ...core.model_trainer import ModelTrainer
from ...core.utils import log_exceptions
from .config import YOLOTrainerConfig
from .model import YOLOModelRadar

if TYPE_CHECKING:
    from ...context.model_trainer_context import ModelContext

torch.multiprocessing.set_sharing_strategy("file_system")


class YOLOTrainerRadar(ModelTrainer):
    config_class = YOLOTrainerConfig
    config: YOLOTrainerConfig

    def __init__(self, config: YOLOTrainerConfig, context: ModelContext) -> None:
        super().__init__(config, context)

        logger.info(f"Model_dir {self.context.model_dir}")

        if self.config.test_dir is not None:
            msg = f"Test Path {self.config.test_dir} provided does not exist"
            assert self.config.test_dir.exists(), msg

        logger.info("YOLO RADAR TRAINER CALLED")

    def load_model(self, path: Path, version: int) -> YOLOModelRadar:
        self.context.stop_model()
        logger.info(f" Trainer Loading from path {path}")
        self.prev_model_path = path
        return YOLOModelRadar(self.config, self.context, path, version)

    @log_exceptions
    def train_model(self, train_dir: Path) -> YOLOModelRadar:
        new_version = self.get_new_version()

        model_savepath = self.context.model_path(new_version, template="model-{}.pt")

        num_classes = len(self.context.class_list)
        # change range to 0 to also train on empty (no detection) images in 0/
        # directory from examples dir
        labels = [str(label) for label in range(1, num_classes)]

        trainpath = self.context.model_path(new_version, template="train-{}.txt")
        train_len = self.make_train_txt(
            trainpath, train_dir, labels, include_label=False
        )

        if self.config.test_dir is not None:
            valpath = self.context.model_path(new_version, template="val-{}.txt")
            val_dir = self.config.test_dir
            self.make_train_txt(valpath, val_dir, labels, include_label=False)

        num_epochs = self.get_num_epochs(version=new_version, positives=train_len["1"])

        data_dict = {  # need to modify this data dict
            "path": str(self.context.model_dir),
            "train": str(trainpath),
            "nc": 1,
            "names": ["positive"],
        }
        if self.config.test_dir is not None:
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
            "hawk.scout.trainer.yolo_radar.yolov5_radar.train",
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
        subprocess.run(cmd, check=True)

        if not model_savepath.exists():
            raise FileNotFoundError
        logger.info("Training completed")

        self.context.stop_model()

        self.prev_model_path = model_savepath
        return YOLOModelRadar(
            self.config,
            self.context,
            model_savepath,
            new_version,
            train_examples=train_len,
        )
