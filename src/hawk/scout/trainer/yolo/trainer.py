# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import glob
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import yaml
from logzero import logger

from ...context.model_trainer_context import ModelContext
from ...core.model import Model
from ...core.model_trainer import ModelTrainerBase
from ...core.utils import log_exceptions
from .model import YOLOModel

torch.multiprocessing.set_sharing_strategy("file_system")


class YOLOTrainer(ModelTrainerBase):
    def __init__(self, context: ModelContext, args: Dict[str, str]):
        super().__init__(args)

        self.args["test_dir"] = self.args.get("test_dir", "")
        self.args["batch-size"] = int(self.args.get("batch-size", 16))
        self.args["image-size"] = int(self.args.get("image-size", 640))
        self.args["initial_model_epochs"] = int(
            self.args.get("initial_model_epochs", 30)
        )
        self.train_initial_model = False

        self.context = context

        logger.info(f"Model_dir {self.context.model_dir}")

        if self.args["test_dir"]:
            self.test_dir = Path(self.args["test_dir"])
            msg = f"Test Path {self.test_dir} provided does not exist"
            assert self.test_dir.exists(), msg

        if self.args["mode"] == "notional":
            assert "notional_model_path" in self.args, "Missing keyword {}".format(
                "notional_model_path"
            )
            notional_model_path = Path(self.args["notional_model_path"])
            msg = f"Notional Model Path {notional_model_path} provided does not exist"
            assert notional_model_path.exists(), msg

        logger.info("YOLO TRAINER CALLED")

    @log_exceptions
    def load_model(self, path: Path = "", content: bytes = b"", version: int = -1):
        if isinstance(path, str):
            path = Path(path)

        if version == -1:
            version = self.get_new_version()

        if self.args["mode"] != "oracle":
            assert path.is_file() or len(content)
            if not path.is_file():
                path = self.context.model_path(version, template="model-{}.pt")
                path.write_bytes(content)

        self.prev_path = path
        self.context.stop_model()
        logger.info(f" Trainer Loading from path {path}")
        return YOLOModel(
            self.args, path, version, mode=self.args["mode"], context=self.context
        )

    @log_exceptions
    def train_model(self, train_dir) -> Model:
        # check mode if not hawk return model
        # EXPERIMENTAL
        if self.args["mode"] == "oracle":
            return self.load_model(Path(""), version=0)
        elif self.args["mode"] == "notional":
            notional_path = self.args["notional_model_path"]
            # sleep for training time
            time_sleep = self.args.get("notional_train_time", 0)
            time_now = time.time()
            while (time.time() - time_now) < time_sleep:
                time.sleep(1)

            return self.load_model(Path(notional_path), version=0)

        new_version = self.get_new_version()

        model_savepath = self.context.model_path(new_version, template="model-{}.pt")
        trainpath = self.context.model_path(new_version, template="train-{}.txt")

        labels = ["1"]
        train_samples = {l: glob.glob(str(train_dir / l / "*")) for l in labels}
        train_len = {l: len(train_samples[l]) for l in labels}
        if train_len["1"] == 0:
            logger.error(train_len)
            logger.error([str(train_dir / l / "*") for l in labels])
            raise Exception

        with open(trainpath, "w") as f:
            for l in labels:
                for path in train_samples[l]:
                    f.write(f"{path}\n")

        noval = True
        if self.args["test_dir"]:
            noval = False
            valpath = self.context.model_path(new_version, template="val-{}.txt")
            with open(valpath, "w") as f:
                for path in glob.glob(self.args["test_dir"] + "/*/*"):
                    f.write(f"{path}\n")

        if new_version <= 0:
            self.train_initial_model = True
            num_epochs = self.args["initial_model_epochs"]
        else:
            online_epochs = json.loads(self.args["online_epochs"])

            if isinstance(online_epochs, list):
                for epoch, pos in online_epochs:
                    pos = int(pos)
                    epoch = int(epoch)
                    if train_len["1"] >= pos:
                        num_epochs = epoch
                        break
            else:
                num_epochs = int(online_epochs)

        file_path = os.path.dirname(os.path.abspath(__file__))

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

        if self.train_initial_model:
            weights = "yolov5s.pt"
        else:
            weights = self.prev_path

        cmd = [
            sys.executable,
            f"{file_path}/yolov5/train.py",
            "--savepath",
            str(model_savepath),
            "--epochs",
            str(num_epochs),
            "--batch-size",
            str(self.args["batch-size"]),
            "--weights",
            str(weights),
            "--imgsz",
            str(self.args["image-size"]),
            "--data",
            str(data_file),
        ]
        if self.args["test_dir"]:
            cmd.extend(["--noval", "False"])

        logger.info(f"TRAIN CMD \n {shlex.join(cmd)}")
        proc = subprocess.Popen(cmd)
        proc.communicate()
        if not model_savepath.exists():
            raise FileNotFoundError
        logger.info("Training completed")

        self.prev_path = model_savepath

        model_args = self.args.copy()
        model_args["train_examples"] = train_len
        self.context.stop_model()
        return YOLOModel(
            model_args,
            model_savepath,
            new_version,
            self.args["mode"],
            context=self.context,
        )
