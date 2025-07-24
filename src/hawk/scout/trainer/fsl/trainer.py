# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import base64
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import torch
from logzero import logger
from torch import optim
from torchvision import models, transforms

from ...core.model_trainer import ModelTrainer
from .config import FSLTrainerConfig
from .model import FSLModel
from .utils import TripletData, TripletLoss

if TYPE_CHECKING:
    from pathlib import Path

    from ...context.model_trainer_context import ModelContext

torch.multiprocessing.set_sharing_strategy("file_system")
device = "cuda" if torch.cuda.is_available() else "cpu"


class FSLTrainer(ModelTrainer):
    config_class = FSLTrainerConfig
    config: FSLTrainerConfig

    def __init__(self, config: FSLTrainerConfig, context: ModelContext) -> None:
        super().__init__(config, context)

        # FIXME use tmpfile
        assert self.config.support_data is not None
        support_data = base64.b64decode(self.config.support_data)
        self.config.support_path.write_bytes(support_data)

        logger.info(f"Model_dir {self.context.model_dir}")

        logger.info("FSL TRAINER CALLED")

    def load_model(self, path: Path, version: int) -> FSLModel:
        return FSLModel(self.config, self.context, path, version)

    def train_model(self, train_dir: Path) -> FSLModel:
        new_version = self.get_new_version()
        model_savepath = self.context.model_path(new_version)

        train_dataset = self.config.fsl_traindir
        support_path = self.config.support_path

        cmd = [
            sys.executable,
            "-m",
            "hawk.scout.trainer.fsl.augment",
            str(train_dataset),
            str(support_path),
        ]
        proc = subprocess.Popen(cmd)
        proc.communicate()

        train_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ],
        )

        # val_transforms = transforms.Compose(
        #     [
        #         transforms.Resize((224, 224)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        #         ),
        #     ]
        # )

        train_data = TripletData(train_dataset, train_transforms)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=32,
            shuffle=True,
            num_workers=4,
        )

        device = "cuda"

        # Our base model
        model = models.resnet18().cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        triplet_loss = TripletLoss()

        # Training
        time_start = time.time()
        epochs = 2
        for epoch in range(epochs):
            model.train()
            epoch_loss = torch.Tensor(0.0)
            for data in train_loader:
                optimizer.zero_grad()
                x1, x2, x3 = data
                e1 = model(x1.to(device))
                e2 = model(x2.to(device))
                e3 = model(x3.to(device))

                loss = triplet_loss(e1, e2, e3)
                epoch_loss += loss
                loss.backward()
                optimizer.step()
            logger.info(f"Train Loss: {epoch} {epoch_loss.item()}")

        time_end = time.time()
        logger.info(f"Training time = {time_end - time_start}")

        torch.save(model.state_dict(), model_savepath)
        # train completed time
        train_time = time_end - time_start

        self.prev_model_path = model_savepath
        return FSLModel(
            self.config,
            self.context,
            model_savepath,
            new_version,
            train_examples={"1": 1, "0": 0},
            train_time=train_time,
        )
