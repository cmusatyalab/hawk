# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import glob
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from logzero import logger

from ...context.model_trainer_context import ModelContext
from ...core.model_trainer import ModelTrainerBase
from ...core.utils import log_exceptions
from .model import YOLOModel

torch.multiprocessing.set_sharing_strategy("file_system")


class YOLOTrainer(ModelTrainerBase):
    def __init__(self, context: ModelContext, args: Dict[str, str]):
        super().__init__(context, args)

        self.args["test_dir"] = self.args.get("test_dir", "")
        self.args["batch-size"] = int(self.args.get("batch-size", 16))
        self.args["image-size"] = int(self.args.get("image-size", 640))
        self.args["initial_model_epochs"] = int(
            self.args.get("initial_model_epochs", 30)
        )
        self.train_initial_model = False

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
    def load_model(
        self, path: Optional[Path] = None, content: bytes = b"", version: int = -1
    ) -> YOLOModel:
        if version == -1:
            version = self.get_new_version()

        if self.args["mode"] != "oracle" and (path is None or not path.is_file()):
            assert len(content)
            path = self.context.model_path(version, template="model-{}.pt")
            path.write_bytes(content)

        assert path is not None

        self.prev_path = path
        self.context.stop_model()
        logger.info(f" Trainer Loading from path {path}")
        return YOLOModel(
            self.args, path, version, mode=self.args["mode"], context=self.context
        )

    @log_exceptions
    def train_model(self, train_dir: Path) -> YOLOModel:
        # check mode if not hawk return model
        # EXPERIMENTAL
        if self.args["mode"] == "oracle":
            return self.load_model(version=0)
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
                for path in train_samples[label]:
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

        weights = Path("yolov5s.pt") if self.train_initial_model else self.prev_path

        cmd = [
            sys.executable,
            "-m",
            "hawk.scout.trainer.yolo.yolov5.train",
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
            # "--freeze", ## if wanting to freeze backbone
            # str(10),
        ]
        capture_files = [data_file, trainpath, train_dir]

        # if not self.train_initial_model:
        #     capture_files.append(weights)

        if self.args["test_dir"]:
            cmd.extend(["--noval", "False"])
            capture_files.append(valpath)

        cmd_str = shlex.join(cmd)
        self.capture_trainingset(cmd_str, capture_files)

        logger.info(f"TRAIN CMD \n {cmd_str}")
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
