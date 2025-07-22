# SPDX-FileCopyrightText: 2022-2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from logzero import logger

from ...context.model_trainer_context import ModelContext
from ...core.model_trainer import ModelTrainerBase
from .model import ActivityClassifierModel

torch.multiprocessing.set_sharing_strategy("file_system")


class ActivityTrainer(ModelTrainerBase):
    def __init__(self, context: ModelContext, args: Dict[str, str]):
        super().__init__(context, args)

        # self.args["test_dir"] = self.args.get("test_dir", "")
        # self.args["arch"] = self.args.get("arch", "resnet50")
        # self.args["batch-size"] = int(self.args.get("batch-size", 64))
        # self.args["unfreeze"] = int(self.args.get("unfreeze_layers", 0))
        # self.args["initial_model_epochs"] = int(
        #     self.args.get("initial_model_epochs", 15)
        # )
        self.train_initial_model = False
        # self.testpath = self.args["test_dir"]

        logger.info(f"Model_dir {self.context.model_dir}")
        #
        # if self.args["test_dir"]:
        #     self.test_dir = Path(self.args["test_dir"])
        #     msg = f"Test Path {self.test_dir} provided does not exist"
        #     assert self.test_dir.exists(), msg

        logger.info("DNN CLASSIFIER TRAINER CALLED")

    def load_model(
        self, path: Optional[Path] = None, content: bytes = b"", version: int = -1
    ) -> ActivityClassifierModel:
        new_version = self.get_new_version()

        if path is None or not path.is_file():
            assert len(content)
            path = self.context.model_path(new_version)
            path.write_bytes(content)

        version = self.get_version()
        logger.info(f"Loading from path {path}")
        self.prev_path = path
        return ActivityClassifierModel(
            self.args, path, version, mode=self.args["mode"], context=self.context
        )

    def train_model(self, train_dir: Path) -> ActivityClassifierModel:
        # check mode if not hawk return model
        # EXPERIMENTAL
        if self.args["mode"] == "oracle":
            return self.load_model(self.prev_path, version=0)
        elif self.args["mode"] == "notional":
            # notional_path = self.args['notional_model_path']
            notional_path = self.prev_path
            # sleep for training time
            time_sleep = self.args.get("notional_train_time", 0)
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

            self.args["test_dir"] = str(valpath)

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
        cmd = [
            sys.executable,
            "-m",
            "hawk.scout.trainer.k600_classifier.train_model",
            "--trainpath",
            str(trainpath),
            "--savepath",
            str(model_savepath),
            "--epochs",
            str(num_epochs),
            "--batch_size",
            str(self.args["batch-size"]),
            "--embed_dim",
            self.args["embed_dim"],
            "--T",
            self.args["T"],
            "--depth",
            self.args["depth"],
            "--num_heads",
            self.args["num_heads"],
            "--mlp_dim",
            self.args["mlp_dim"],
            "--num_classes",
            self.args["num_classes"],
            "--head_dim",
            self.args["head_dim"],
        ]
        capture_files = [trainpath, train_dir]

        if self.train_initial_model:
            self.train_initial_model = False
        else:
            cmd.extend(["--resume", str(self.prev_path)])
            # capture_files.append(self.prev_path)

        if self.args["test_dir"]:
            cmd.extend(["--valpath", self.args["test_dir"]])
            capture_files.extend([valpath, val_dir])

        cmd_str = shlex.join(cmd)
        self.capture_trainingset(cmd_str, capture_files)

        logger.info(f"TRAIN CMD\n {cmd_str}")
        proc = subprocess.Popen(cmd)
        proc.communicate()

        # train completed time
        train_time = time.time() - self.context.start_time

        self.prev_path = model_savepath

        model_args = self.args.copy()
        model_args["train_examples"] = train_len
        model_args["train_time"] = train_time

        return ActivityClassifierModel(
            model_args,
            model_savepath,
            new_version,
            self.args["mode"],
            context=self.context,
        )
