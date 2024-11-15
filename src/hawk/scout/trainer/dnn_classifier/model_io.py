# SPDX-FileCopyrightText: 2022-2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import torch
from torchvision import models


class CheckpointState(TypedDict):
    arch: str
    num_classes: int
    epoch: int
    state_dict: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]


def load_checkpoint_state(
    checkpoint_path: Path | None, arch: str, num_classes: int
) -> CheckpointState:
    if checkpoint_path is not None and checkpoint_path.is_file():
        checkpoint: CheckpointState = torch.load(checkpoint_path)
    else:
        checkpoint = {
            "arch": arch,
            "num_classes": num_classes,
            "epoch": 0,
            "state_dict": {},
            "optimizer": {},
            "scheduler": {},
        }
    if checkpoint["arch"] != arch:
        msg = "Loaded checkpoint has unexpected architecture {checkpoint_arch}"
        raise ValueError(msg)
    return checkpoint


def save_checkpoint_state(
    checkpoint_path: Path,
    arch: str,
    epoch: int,
    num_classes: int,
    model: torch.nn.Module,
    optimizer: torch.nn.Module,
    scheduler: torch.nn.Module,
) -> None:
    checkpoint: CheckpointState = {
        "arch": arch,
        "num_classes": num_classes,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)


def load_model_for_inference(checkpoint: CheckpointState) -> torch.nn.Module:
    arch = checkpoint["arch"]
    num_classes = checkpoint["num_classes"]

    model = models.__dict__[arch](pretrained=True)
    patch_model(model, arch, num_classes)

    # load weights from checkpoint
    if checkpoint["state_dict"]:
        model.load_state_dict(checkpoint["state_dict"])

    # if torch.cuda.is_available():
    #     print("using GPU")
    #     model = model.cuda()
    # else:
    #     print("using CPU, this will be slow")
    return model


def load_model_for_training(
    # model settings
    checkpoint: CheckpointState,
    num_classes: int,
    num_unfreeze: int,
    # optimizer settings
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    # scheduler settings
    warmup_epochs: int,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    model = load_model_for_inference(checkpoint)

    # replace last layer if we are retraining and number of classes has changed
    new_final_layer = num_classes != checkpoint["num_classes"]
    if new_final_layer:
        arch = checkpoint["arch"]
        patch_model(model, arch, num_classes)

    freeze_layers(model, num_unfreeze)

    if torch.cuda.is_available():
        print("using GPU")
        model = model.cuda()
    else:
        print("using CPU, this will be slow")

    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    if not new_final_layer and checkpoint["optimizer"]:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
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

    return model, optimizer, scheduler


def patch_model(model: torch.nn.Module, arch: str, num_classes: int) -> None:
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


def freeze_layers(model: torch.nn.Module, unfreeze: int = -1) -> None:
    if unfreeze == -1:
        return

    unfreeze += 1  # the avg pool layer isn't trainable

    len_layers = len(list(model.children()))
    num_freeze = len_layers - unfreeze

    for count, child in enumerate(model.children()):
        if count >= num_freeze:
            break
        for param in child.parameters():
            param.requires_grad = False


def model_input_size(arch: str) -> int:
    if "inception" in arch:
        return 299
    return 224
