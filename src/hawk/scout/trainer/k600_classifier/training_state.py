# SPDX-FileCopyrightText: 2022-2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import torch
from torchvision.models import get_model, get_model_weights
from torchvision.transforms.v2 import Transform


class CheckpointState(TypedDict):
    """Representation of the state written to a checkpoint file.

    arch is the name of the torchvision model.
    weights is the name of the model specific weights (IMAGENET1K_V1 or _V2).
    num_classes is the number of classes in the dataset.
    epoch is the epoch the model was trained to.
    state_dict is the state dict of the model.
    optimizer is the optimizer state dict.
    scheduler is the scheduler state dict.

    weights and num_classes may not exist in older save files, they default to
    weights=IMAGENET1K_V1 and num_classes=2.
    """

    arch: str
    weights: str
    num_classes: int
    epoch: int
    state_dict: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]


@dataclass
class TrainingState:
    arch: str
    weights: str
    num_classes: int
    torch_model: torch.nn.Module
    preprocess: Transform
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    epoch: int = 0

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        """Save the training state to a checkpoint file."""
        checkpoint_state: CheckpointState = {
            "arch": self.arch,
            "weights": self.weights,
            "num_classes": self.num_classes,
            "epoch": self.epoch,
            "state_dict": self.torch_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(checkpoint_state, checkpoint_path)

    @staticmethod
    def _load_checkpoint_state(
        checkpoint_path: Path, bootstrap_arch: str
    ) -> CheckpointState:
        """Load the training state from a checkpoint file."""
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint {checkpoint_path} does not exist")

        checkpoint_state: CheckpointState = torch.load(checkpoint_path)

        # stay backward compatible with older bootstrap models.
        checkpoint_state.setdefault("arch", bootstrap_arch)
        checkpoint_state.setdefault("num_classes", 2)
        checkpoint_state.setdefault("weights", "IMAGENET1K_V1")
        return checkpoint_state

    @classmethod
    def _load_model_from_checkpoint(
        cls, checkpoint: CheckpointState
    ) -> tuple[torch.nn.Module, Transform, int]:
        """Create or load a model based on the parameters in the checkpoint state."""
        arch = checkpoint["arch"]
        bootstrap_weights = checkpoint["weights"]
        weights = get_model_weights(arch).__members__[bootstrap_weights]

        torch_model = get_model(arch, weights=weights)

        num_classes = checkpoint["num_classes"]
        patch_final_layer(torch_model, arch, num_classes)

        if checkpoint["state_dict"]:
            torch_model.load_state_dict(checkpoint["state_dict"])

        preprocess = weights.transforms()
        return torch_model, preprocess, checkpoint["epoch"]

    @classmethod
    def load_for_inference(
        cls, checkpoint_path: Path, bootstrap_arch: str
    ) -> tuple[torch.nn.Module, Transform, int]:
        """Load a torch model for inference from a checkpoint file."""
        checkpoint = cls._load_checkpoint_state(checkpoint_path, bootstrap_arch)
        return cls._load_model_from_checkpoint(checkpoint)

    @classmethod
    def load_for_training(
        cls,
        checkpoint_path: Path | None,
        # model settings
        bootstrap_arch: str,
        bootstrap_weights: str,
        num_classes: int,
        num_unfreeze: int,
        # optimizer settings
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        # scheduler settings
        warmup_epochs: int,
    ) -> TrainingState:
        """Create a model and training state based on a checkpoint file.
        If the checkpoint file does not exist, we initialize a new state based
        on the specified bootstrap_arch/bootstrap_weights/num_classes parameters.
        """
        try:
            if checkpoint_path is None:
                raise ValueError
            checkpoint = cls._load_checkpoint_state(checkpoint_path, bootstrap_arch)
        except ValueError:
            checkpoint = {
                "arch": bootstrap_arch,
                "weights": bootstrap_weights,
                "num_classes": num_classes,
                "epoch": 0,
                "state_dict": {},
                "optimizer": {},
                "scheduler": {},
            }
        arch = checkpoint["arch"]
        torch_model, preprocess, epoch = cls._load_model_from_checkpoint(checkpoint)

        replace_final_layer = checkpoint["num_classes"] != num_classes
        if replace_final_layer:
            patch_final_layer(torch_model, arch, num_classes)

        freeze_layers(torch_model, num_unfreeze)
        torch_model = to_gpu(torch_model)

        # load optimizer
        optimizer = torch.optim.SGD(
            torch_model.parameters(),
            learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        if not replace_final_layer and checkpoint["optimizer"]:
            optimizer.load_state_dict(checkpoint["optimizer"])

        # load scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=100, gamma=0.9
        )
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.5, total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, lr_scheduler],
            milestones=[warmup_epochs],
        )
        if checkpoint["scheduler"]:
            scheduler.load_state_dict(checkpoint["scheduler"])

        return cls(
            arch=arch,
            weights=checkpoint["weights"],
            num_classes=num_classes,
            torch_model=torch_model,
            preprocess=preprocess,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=checkpoint["epoch"],
        )

    @property
    def input_size(self) -> int:
        """Return the input size of the model."""
        return 224 if "inception" not in self.arch else 299

    @property
    def is_resumed(self) -> bool:
        """Return True if the training state has been resumed from a checkpoint."""
        return self.epoch != 0

    def rollback(self, checkpoint_path: Path) -> None:
        """Rollback the model to a previous checkpoint."""
        if checkpoint_path is None or not checkpoint_path.is_file():
            return

        checkpoint = self._load_checkpoint_state(checkpoint_path, self.arch)

        self.torch_model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

    def merge(self, checkpoint_path: Path, alpha: float) -> None:
        """Exponential Moving Average merge with a previous checkpoint."""
        if checkpoint_path is None or not checkpoint_path.is_file() or alpha == 0.0:
            return

        checkpoint = self._load_checkpoint_state(checkpoint_path, self.arch)

        assert checkpoint["arch"] == self.arch
        if checkpoint["num_classes"] != self.num_classes:
            return

        old_model = checkpoint["state_dict"]
        cur_model = self.torch_model.state_dict()

        neg_alpha = 1.0 - alpha
        for key in old_model:
            cur_model[key] = neg_alpha * cur_model[key] + alpha * old_model[key]

        self.torch_model.load_state_dict(cur_model)


def patch_final_layer(model: torch.nn.Module, arch: str, num_classes: int) -> None:
    """Patch the final layer of the model for the specified number of classes."""
    if "resnet" in arch:
        """Resnet"""
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)

    elif "alexnet" in arch:
        """Alexnet"""
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)

    elif "vgg" in arch:
        """VGG11_bn"""
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)

    elif "squeezenet" in arch:
        """Squeezenet"""
        model.classifier[1] = torch.nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        model.num_classes = num_classes

    elif "densenet" in arch:
        """Densenet"""
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, num_classes)

    elif "inception" in arch:
        """Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        # Handle the auxilary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = torch.nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)

    elif "efficientnet" in arch:
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)

    else:
        msg = f"Invalid model name {arch}"
        raise ValueError(msg)


def freeze_layers(torch_model: torch.nn.Module, unfreeze: int = -1) -> None:
    """Freeze layers of the model except for the last 'unfreeze' number of layers."""
    if unfreeze == -1:
        return

    # the avg pool layer before the final layer isn't trainable which is
    # why we add 1 to the requested number of unfrozen layers here
    unfreeze += 1

    len_layers = len(list(torch_model.children()))
    num_freeze = len_layers - unfreeze

    for count, child in enumerate(torch_model.children()):
        if count >= num_freeze:
            break
        for param in child.parameters():
            param.requires_grad = False


def to_gpu(torch_model: torch.nn.Module) -> torch.nn.Module:
    """Move the model to the GPU if available."""
    if torch.cuda.is_available():
        print("using GPU")
        return torch_model.cuda()
    else:
        print("using CPU, this will be slow")
        return torch_model
