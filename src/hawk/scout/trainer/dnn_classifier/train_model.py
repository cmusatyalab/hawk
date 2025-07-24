# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only
# Source: https://github.com/pytorch/examples/blob/main/imagenet/main.py

from __future__ import annotations

import argparse
import os
import random
import time
import warnings
from enum import Enum
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms.v2 as transforms
from logzero import logger
from sklearn.metrics import (  # auc, precision_recall_curve, roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.backends import cudnn
from torchvision import models
from tqdm import tqdm

from ...core.utils import ImageFromList
from .training_state import TrainingState

model_names = models.list_models()

#
# Shilpa's original experiment had the following arguments as defaults
# --arch resnet50
# --bootstrap-weights IMAGENET1K_V1
# --ema 0.5
# RandomResizedCrop instead of RandomRotation transform
#

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
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "--bootstrap-weights",
    metavar="WEIGHTS",
    default="IMAGENET1K_V1",
    # default="DEFAULT",
    help="model specific weights (DEFAULT|IMAGENET1K_V1|IMAGENET1K_V2)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)

parser.add_argument(
    "-w",
    "--write_scores",
    dest="write_scores",
    default=None,
    help="file path to write all validation scores",
)

parser.add_argument(
    "--num-classes",
    default=2,
    type=int,
    help="number of classes to train",
)
parser.add_argument(
    "--num-unfreeze",
    default=0,
    type=int,
    help="number of layers to train",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs",
    default=10000,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "--warmup-epochs",
    default=5,
    type=int,
    help="initial number of epochs for warmup",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--break-epoch",
    default=-1,
    type=int,
    help="break epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=64,
    type=int,
    help="mini-batch size (default: 32), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.01,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--min-lr", default=1e-4, type=float, help="minimum learning rate")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "--resume",
    default=None,
    type=Path,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="seed for initializing training. ",
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--ema",
    type=float,
    # default=0.5,
    default=0.2,
    help="average with last checkpoint (0 is no averaging, default 0.2)",
)

best_acc1 = 0.0


def main() -> None:
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints.",
            stacklevel=1,
        )

    ngpus_per_node = torch.cuda.device_count()

    if args.evaluate:
        assert os.path.exists(args.valpath)
        eval_worker(args.gpu, ngpus_per_node, args)
    else:
        assert os.path.exists(args.trainpath)
        train_worker(args.gpu, ngpus_per_node, args)


def write_scores(file_path: Path, y_pred: list[int], y_true: list[int]) -> None:
    if os.path.exists(file_path):  ## remove if already exists
        os.remove(file_path)
    with open(file_path, "w") as f:
        f.writelines(
            f"{pred:0.4f} {y_true[i]}\n" for i, pred in enumerate(y_pred)
        )  ## write all predictions and labels to each line


def eval_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace) -> None:
    args.gpu = gpu
    # start_time = time.time()
    print("GPU:", args.gpu)

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    # load model from checkpoint
    print(f"=> loading checkpoint state '{args.resume}'")
    torch_model, preprocess, args.start_epoch = TrainingState.load_for_inference(
        args.resume,
        args.arch,
    )
    if args.start_epoch:
        print(f"=> loaded checkpoint '{args.resume}'")
    elif args.resume:
        print(f"=> no checkpoint found at '{args.resume}'")

    if args.gpu:
        torch_model = torch_model.cuda()
    else:
        torch_model = torch_model.cuda()
        print("Not triggering GPU load for model")
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    val_path = args.valpath
    logger.info(f"Test path {val_path}")
    val_list = []
    val_labels = []

    with open(val_path) as f:
        contents = f.read().splitlines()
        for content in contents:
            path, label = content.split()
            val_list.append(Path(path))
            val_labels.append(int(label))

    val_dataset = ImageFromList(
        val_list,
        preprocess,
        label_list=val_labels,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    auc = validate_model(torch_model, val_loader, criterion, args)

    logger.info(f"Model AUC {auc}")


def train_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace) -> None:
    global best_acc1
    args.gpu = gpu
    start_time = time.time()

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    if args.savepath is None:
        args.savepath = Path("checkpoint.pth")

    print(f"=> loading checkpoint state '{args.resume}'")
    model_state = TrainingState.load_for_training(
        args.resume,
        bootstrap_arch=args.arch,
        bootstrap_weights=args.bootstrap_weights,
        num_classes=args.num_classes,
        num_unfreeze=args.num_unfreeze,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
    )

    if model_state.is_resumed:
        print(f"=> loaded checkpoint '{args.resume}' (epoch {model_state.epoch})")
        args.start_epoch = model_state.epoch
    elif args.resume:
        print(f"=> no checkpoint found at '{args.resume}'")

    cudnn.benchmark = True

    # Data loading code
    train_path = args.trainpath
    train_list = []
    train_labels = []
    with open(train_path) as f:
        contents = f.read().splitlines()
        for content in contents:
            path, label = content.split()
            train_list.append(Path(path))
            train_labels.append(int(label))

    preprocess = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.uint8, scale=True),
            # transforms.RandomResizedCrop(model_state.input_size),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToDtype(torch.float32, scale=True),
            model_state.preprocess,
        ],
    )
    train_dataset = ImageFromList(
        train_list,
        preprocess,
        label_list=train_labels,
        limit=500 * sum(train_labels),
    )

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    args.validate = bool(args.valpath)
    logger.info(f"Validate {args.validate}")
    if args.validate:
        val_path = args.valpath
        logger.info(f"Test path {val_path}")
        val_list = []
        val_labels = []

        with open(val_path) as f:
            contents = f.read().splitlines()
            for content in contents:
                path, label = content.split()
                val_list.append(Path(path))
                val_labels.append(int(label))

        val_dataset = ImageFromList(
            val_list,
            model_state.preprocess,
            label_list=val_labels,
            limit=500 * sum(val_labels),
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

    targets = torch.LongTensor(train_dataset.targets)
    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)],
    )
    logger.info(f"Class sample count: {class_sample_count}")
    total_samples = sum(class_sample_count)
    class_weights = [1 - (float(x) / float(total_samples)) for x in class_sample_count]
    # class_weight_neg = class_weights[0]/2
    # class_weight_pos = 1 - class_weight_neg
    # class_weights = [class_weight_neg, class_weight_pos]
    # class_weights = [1/21,20/21]

    logger.info(f"Total samples {total_samples} Class Weight {class_weights}")
    criterion = nn.CrossEntropyLoss(
        weight=torch.Tensor(class_weights),
        label_smoothing=0.1,
    ).cuda()

    break_epoch = (
        args.epochs if args.break_epoch == -1 else args.start_epoch + args.break_epoch
    )

    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_seed()

        # train for one epoch
        train(model_state, train_loader, criterion)
        model_state.epoch = epoch + 1

        logger.info(f"Epoch {epoch}")
        if args.validate:
            # evaluate on validation set
            acc1 = validate_model(model_state.torch_model, val_loader, criterion, args)
            # remember best acc@1 and save checkpoint
            is_best = acc1 >= best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                logger.info(f"Saving model AUC: {best_acc1}")
                model_state.save_checkpoint(args.savepath)

        adjust_learning_rate(model_state, epoch, args)

        if epoch >= break_epoch:
            if not args.validate:
                logger.info("Saving last model")
                model_state.save_checkpoint(args.savepath)
            break

    end_time = time.time()
    print(end_time - start_time)

    if args.validate:
        # if we were not using validate, we saved the last epoch which happens
        # to be the current model, so rollback is not necessary
        model_state.rollback(args.savepath)

    # EMA: Averaging models, merge with previous model to provide stability and
    # reduce the effect of human labeling errors
    model_state.merge(args.resume, alpha=args.ema)

    if args.validate:
        best_auc = validate_model(model_state.torch_model, val_loader, criterion, args)
        logger.info(f"Best TEST AUC {best_auc}")

    model_state.save_checkpoint(args.savepath)


def train(
    model_state: TrainingState,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn._Loss,
) -> None:
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")

    # switch to train mode
    model_state.torch_model.train()

    end = time.time()
    for images, target in train_loader:
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = model_state.torch_model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))

        # compute gradient and do SGD step
        assert model_state.optimizer is not None
        model_state.optimizer.zero_grad()
        loss.backward()
        model_state.optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


def adjust_learning_rate(
    model_state: TrainingState,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    # lr = args.lr * (0.1 ** (epoch // 20))
    # for param_group in optimizer.param_groups:
    # param_group['lr'] = lr
    try:
        last_lr = model_state.scheduler.get_last_lr()[0]
    except Exception:
        last_lr = model_state.optimizer.param_groups[0]["lr"]
    if epoch > args.warmup_epochs and last_lr <= args.min_lr:
        return
    model_state.scheduler.step()


def calculate_performance(
    y_true: Sequence[int],
    y_pred: Sequence[Sequence[float]],
) -> float:
    if len(y_pred[0]) > 2:
        y_true_bin = label_binarize(y_true, classes=list(range(len(y_pred[0]))))
        ap_by_class: Sequence[float] = average_precision_score(
            y_true_bin,
            y_pred,
            average=None,
        )
        ap: float = sum(ap_by_class[1:]) / len(ap_by_class[1:])
        logger.info(f" AP by class: {ap_by_class}")
    else:
        ap = average_precision_score(y_true, np.array(y_pred)[:, 1], average=None)
    # roc_auc = roc_auc_score(y_true, y_pred)
    # precision, recall, _ = precision_recall_curve(y_true, y_pred)
    # pr_auc = auc(recall, precision)

    logger.info(f"AUC {ap}")
    # logger.info(f"ROC AUC {roc_auc}")
    # logger.info(f"PR AUC {pr_auc}")
    return ap


def validate_model(
    torch_model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn._Loss,
    args: argparse.Namespace,
) -> float:
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.NONE)

    # switch to evaluate mode
    torch_model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        end = time.time()
        for images, target in tqdm(val_loader):
            if len(images) == 1:
                continue
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = torch_model(images)

            loss = criterion(output, target)

            probability = torch.nn.functional.softmax(output, dim=1)
            # logger.info(f"Prob: {probability}")
            probability = np.squeeze(probability.cpu().numpy())
            try:
                # probability = probability[:, 1:] ## retain
                # logger.info(f"Probs: {probability}")
                y_pred.extend(probability)
                y_true.extend(target.cpu())
            except Exception:
                probability = probability[1]
                y_pred.append(probability)
                y_true.append(target.cpu()[0])

            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        ## here write all samples to file if write scores is a valid file path
        if args.write_scores is not None:
            assert os.path.exists(os.path.dirname(args.write_scores))
            write_scores(
                args.write_scores,
                [pred[1] for pred in y_pred],
                [label.item() for label in y_true],
            )

    return calculate_performance(y_true, y_pred)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(
        self,
        name: str,
        fmt: str = ":f",
        summary_type: Summary = Summary.AVERAGE,
    ) -> None:
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self) -> str:
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            msg = f"invalid summary type {self.summary_type!r}"
            raise ValueError(msg)

        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":
    main()
