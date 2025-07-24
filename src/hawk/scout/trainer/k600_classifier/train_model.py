# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only
# Source: https://github.com/pytorch/examples/blob/main/imagenet/main.py

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms.v2 as transforms
from logzero import logger
from torch import nn
from torch.utils.data import DataLoader

from hawk.scout.trainer.k600_classifier.action_recognition_model import (
    ActionRecognitionModel,
)
from hawk.scout.trainer.k600_classifier.movinet_a0s_encoder import MovinetEncoder
from hawk.scout.trainer.k600_classifier.temporal_encoder import TransformerParams
from hawk.scout.trainer.k600_classifier.tensor_ds import PTListDataset

if TYPE_CHECKING:
    from torch.optim import Optimizer

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--trainpath", type=str, default="", help="path to train file")
parser.add_argument("--valpath", type=str, default="", help="path to val file")
parser.add_argument(
    "--savepath",
    type=Path,
    default=Path("model.pth"),
    help="path to save trained model",
)
parser.add_argument(
    "--num_classes",
    default=2,
    type=int,
    help="number of classes to train",
)
parser.add_argument(
    "--epochs",
    default=10,
    type=int,
    help="number of total epochs to run",
)
parser.add_argument(
    "--batch_size",
    default=10,
    type=int,
    help="batch size (default: 10)",
)
parser.add_argument(
    "--resume",
    default=None,
    type=Path,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument("--embed_dim", type=int, help="embedding dimension")
parser.add_argument("--T", default=5, type=int, help="window (default: 5)")
parser.add_argument("--depth", type=int, help="depth (default: 2)")
parser.add_argument("--num_heads", type=int, help="num_heads (default: 16)")
parser.add_argument("--mlp_dim", type=int, help="batch size (default: 10)")
parser.add_argument("--head_dim", type=int, help="head_dim")


best_acc1 = 0.0


def main() -> None:
    args = parser.parse_args()
    assert os.path.exists(args.trainpath)
    assert os.path.exists(args.valpath)
    train(args)


def train(args: argparse.Namespace) -> None:
    global best_acc1
    start_time = time.time()
    assert args.savepath is not None
    model_path = args.savepath
    batch_size = args.batch_size
    # Data loading code
    train_file_path = args.trainpath
    transform = transforms.Compose(
        [
            lambda v: v.to(torch.float32) / 255,
            transforms.Resize((200, 200)),
            transforms.RandomCrop((172, 172)),
        ],
    )
    train_ds = PTListDataset(train_file_path, transform)
    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
    )

    val_file_path = args.valpath
    logger.info(f"Test path {val_file_path}")
    test_transform = transforms.Compose(
        [
            lambda v: v.to(torch.float32) / 255,
            transforms.Resize((200, 200)),
            transforms.CenterCrop((172, 172)),
        ],
    )
    val_ds = PTListDataset(train_file_path, test_transform)
    val_dataloader = DataLoader(
        val_ds,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
    )

    embed_dim = int(args.embed_dim)
    T = int(args.T)
    depth = int(args.depth)
    num_heads = int(args.num_heads)
    mlp_dim = int(args.mlp_dim)
    num_classes = int(args.num_classes)
    head_dim = int(args.head_dim)
    transformer_params = TransformerParams(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_classes=num_classes,
        head_dim=head_dim,
    )
    model = ActionRecognitionModel(
        MovinetEncoder(embed_dim=transformer_params.embed_dim),
        transformer_params,
        T=T,
        stride=T,
    )
    t_lr = 1e-3
    s_lr = 1e-4
    optimizer = torch.optim.Adam(
        [
            {"params": model._encoder.parameters(), "lr": s_lr},
            {"params": model._ln.parameters(), "lr": s_lr},
            {"params": model._temporal_enc.parameters(), "lr": t_lr},
        ],
    )
    ce_loss_fn_mean = torch.nn.CrossEntropyLoss()
    ce_loss_fn_sum = torch.nn.CrossEntropyLoss(reduction="sum")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device} device")
    model = prepare_model(model)
    model.to(device)
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}\n-------------------------------")
        train_loss = train_epoch(
            dataloader=train_dataloader,
            model=model,
            ce_loss_fn=ce_loss_fn_mean,
            optimizer=optimizer,
            device=device,
            ds_size=len(train_ds),
            batch_size=batch_size,
        )
        validation_loss = validate(
            dataloader=val_dataloader,
            model=model,
            loss_fn=ce_loss_fn_sum,
            device=device,
            batch_size=batch_size,
        )
        logger.info(
            f"[Epoch {epoch + 1}] Train Loss: {train_loss}, "
            f"Validation Loss: {validation_loss:.3f}",
        )
        model.save(model_path, num_samples=0)
    logger.info("Done!")
    end_time = time.time()
    logger.info(end_time - start_time)


def train_epoch(
    dataloader: DataLoader,
    model: ActionRecognitionModel,
    ce_loss_fn: nn.Module,
    optimizer: Optimizer,
    device: str,
    ds_size: int,
    batch_size: int,
) -> float:
    set_train(model)
    optimizer.zero_grad()
    size = ds_size
    acc_loss = 0
    for batch_idx, (X, Y) in enumerate(dataloader):
        X = X.to(device)
        Y = Y.to(device)
        logits, _ = model(X)
        loss = ce_loss_fn(logits, Y)
        acc_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch_idx % 100 == 0:
            loss, current = loss.item(), (batch_idx + 1)
            logger.info(f"loss: {loss:>7f}, [{current * batch_size:>5d}/{size:>5d}]")
    return acc_loss / len(dataloader)


def validate(
    dataloader: DataLoader,
    model: ActionRecognitionModel,
    loss_fn: nn.Module,
    device: str,
    batch_size: int,
) -> float:
    set_val(model)
    validation_loss = 0.0
    N = 0
    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(device)
            Y = Y.to(device)
            logits, _ = model(X)
            loss = loss_fn(logits, Y)
            validation_loss += loss.item()
            N += 1
    return validation_loss / (N * batch_size)


def prepare_model(model: ActionRecognitionModel) -> ActionRecognitionModel:
    for param in model._encoder.parameters():
        param.requires_grad = False
    for param in model._encoder._encoder.classifier.parameters():
        param.requires_grad = True
    for param in model._ln.parameters():
        param.requires_grad = True
    for param in model._temporal_enc.parameters():
        param.requires_grad = True
    return model


def set_train(model: ActionRecognitionModel) -> None:
    # model.train()
    model._temporal_enc.train()
    model._ln.train()
    model._encoder.eval()
    model._encoder._encoder.classifier.train()


def set_val(model: ActionRecognitionModel) -> None:
    model.eval()


if __name__ == "__main__":
    main()
