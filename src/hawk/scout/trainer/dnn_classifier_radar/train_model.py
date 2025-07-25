# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only
# Source: https://github.com/pytorch/examples/blob/main/imagenet/main.py

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from logzero import logger
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR
from torchvision import models, transforms
from tqdm import tqdm

from ...core.utils import ImageFromList

# from sklearn.preprocessing import label_binarize


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--trainpath", type=str, default="", help="path to tain file")
parser.add_argument("--valpath", type=str, default="", help="path to val file")
parser.add_argument(
    "--savepath",
    type=Path,
    default="model.pth",
    help="path to save trained model",
)
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
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
    type=str,
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
    "--base_model_path",
    default=None,
    type=str,
    metavar="PATH",
    help="path to base (foundation) model for transfer learning",
)

best_acc1 = 0.0

########
# Change this to the path on the scout where the base model is located (after
# downloading from Git repo).  Will add this to config file later.
base_model_path = (
    "/srv/diamond/RADAR_DERIVATIVE_CROPPED/person/person_basemodel_resnet18.pth"
)
########


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


def eval_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace) -> None:
    args.gpu = gpu
    # start_time = time.time()

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    print(f"=> using pre-trained model '{args.arch}'")
    model = models.__dict__[args.arch](pretrained=True)
    model, input_size = initialize_model(args.arch, args.num_classes, args.num_unfreeze)

    if not torch.cuda.is_available():
        print("using CPU, this will be slow")
    else:
        print("using GPU")
        model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # load model from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)

            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            print(f"=> loaded checkpoint {args.resume}")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    radar_normalize = transforms.Normalize(
        mean=[0.111, 0.110, 0.111],
        std=[0.052, 0.050, 0.052],
    )

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
        transforms.Compose(
            [
                transforms.Pad(padding=(80, 0), fill=0, padding_mode="constant"),
                transforms.Resize(input_size),
                transforms.ToTensor(),
                radar_normalize,
            ],
        ),
        label_list=val_labels,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    auc = validate_model(val_loader, model, criterion, args)

    logger.info(f"Model AUC {auc}")


def train_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace) -> None:
    global best_acc1
    args.gpu = gpu
    start_time = time.time()

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    print(f"=> using pre-trained model '{args.arch}'")
    model = models.__dict__[args.arch](weights="ResNet18_Weights.DEFAULT")
    logger.info(f" base model path: {args.base_model_path}")
    model, input_size = initialize_model(
        args.arch,
        args.num_classes,
        args.num_unfreeze,
        args.base_model_path,
    )

    if not torch.cuda.is_available():
        print("using CPU, this will be slow")
    else:
        print("using GPU")
        model = model.cuda()

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
    logger.info(f"Length of train labels: {len(train_labels)}")

    radar_normalize = transforms.Normalize(
        mean=[0.111, 0.110, 0.111],
        std=[0.052, 0.050, 0.052],
    )

    train_dataset = ImageFromList(
        train_list,
        transforms.Compose(
            [
                transforms.Pad(padding=(80, 0), fill=0, padding_mode="constant"),
                transforms.Resize(input_size),
                transforms.ToTensor(),
                radar_normalize,
            ],
        ),
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
            transforms.Compose(
                [
                    transforms.Pad(padding=(80, 0), fill=0, padding_mode="constant"),
                    transforms.Resize(input_size),
                    # transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    radar_normalize,
                ],
            ),
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
    # class_weights = [
    #     1 - (float(x) / float(total_samples)) for x in class_sample_count
    # ]
    class_weights_temp = [1 / float(x) for x in class_sample_count]
    class_weights = [x / sum(class_weights_temp) for x in class_weights_temp]

    logger.info(f"Total samples {total_samples} Class Weight {class_weights}")
    criterion = nn.CrossEntropyLoss(
        weight=torch.Tensor(class_weights),
        label_smoothing=0.1,
    ).cuda()

    # define loss function (criterion), optimizer, and learning rate scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    lr_scheduler = StepLR(optimizer, step_size=100, gamma=0.9)
    lr_warmup_epochs = args.warmup_epochs
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.5,
        total_iters=lr_warmup_epochs,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_lr_scheduler, lr_scheduler],
        milestones=[lr_warmup_epochs],
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)

            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume,
                    checkpoint["epoch"],
                ),
            )
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")

    break_epoch = (
        args.epochs if args.break_epoch == -1 else args.start_epoch + args.break_epoch
    )

    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_seed()

        # train for one epoch
        logger.info("Above train...")
        train(train_loader, model, criterion, optimizer, epoch, args)

        logger.info(f"Epoch {epoch}")
        if args.validate:
            # evaluate on validation set
            acc1 = validate_model(val_loader, model, criterion, args)
            # remember best acc@1 and save checkpoint
            is_best = acc1 >= best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                logger.info(f"Saving model AUC: {best_acc1}")
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    args.savepath,
                )

        adjust_learning_rate(optimizer, scheduler, epoch, args)
        if epoch >= break_epoch:
            if not args.validate:
                logger.info("Saving last model")
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    args.savepath,
                )

            break

    end_time = time.time()
    print(end_time - start_time)

    # EMA: Averaging models
    if args.validate:
        best_checkpoint = torch.load(args.savepath)
        curr_model = best_checkpoint["state_dict"]
    else:
        curr_model = model.state_dict()

    if args.resume:
        # curr_model = curr_model.detach().cpu()
        checkpoint = torch.load(args.resume)
        old_model = checkpoint["state_dict"]

        for key in old_model:
            curr_model[key] = (curr_model[key] + old_model[key]) / 2.0

    model.load_state_dict(curr_model)

    if args.validate:
        best_auc = validate_model(val_loader, model, criterion, args)
        logger.info(f"Best TEST AUC {best_auc}")

    save_checkpoint(
        {
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        args.savepath,
    )


def set_parameter_requires_grad(model: nn.Module, unfreeze: int = 0) -> None:
    if unfreeze == -1:
        return

    # the avg pool layer before the final layer isn't trainable which is
    # why we add 1 to the requested number of unfrozen layers here
    unfreeze += 1

    len_layers = len(list(model.children()))
    num_freeze = len_layers - unfreeze

    for count, child in enumerate(model.children()):
        if count >= num_freeze:
            break
        for param in child.parameters():
            param.requires_grad = False


def initialize_model(
    arch: str,
    num_classes: int,
    unfreeze: int = 0,
    base_path: Path | None = None,
) -> tuple[nn.Module, int]:
    model_ft = None
    input_size = 0
    model_ft = models.__dict__[arch](pretrained=True)
    # Radar code
    logger.info("Loading initial radar checkpoint!!!")
    # radar_checkpoint = torch.load(base_model_path)
    logger.info(f"Base path: {base_path}")
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(
        num_ftrs,
        5,
    )  # only for training the binary model from 5 to 2 classes.
    if base_path is not None:
        radar_checkpoint = torch.load(base_path)
        model_ft.load_state_dict(radar_checkpoint["state_dict"])
    ## otherwise use the default resnet18 model (RGB)

    if "resnet" in arch:
        """Resnet"""
        set_parameter_requires_grad(model_ft, unfreeze)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif "alexnet" in arch:
        """Alexnet"""
        set_parameter_requires_grad(model_ft, unfreeze)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif "vgg" in arch:
        """VGG11_bn"""
        set_parameter_requires_grad(model_ft, unfreeze)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif "squeezenet" in arch:
        """Squeezenet"""
        set_parameter_requires_grad(model_ft, unfreeze)
        model_ft.classifier[1] = nn.Conv2d(
            512,
            num_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
        )
        model_ft.num_classes = num_classes
        input_size = 224

    elif "densenet" in arch:
        """Densenet"""
        set_parameter_requires_grad(model_ft, unfreeze)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif "inception" in arch:
        """Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        set_parameter_requires_grad(model_ft, unfreeze)
        # Handle the auxiliary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    elif "efficientnet" in arch:
        set_parameter_requires_grad(model_ft, unfreeze)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        sys.exit()
    logger.info(f"Number of classes: {num_classes}")

    return model_ft, input_size


def train(
    train_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn._Loss,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")

    # switch to train mode
    model.train()

    end = time.time()
    for images, target in tqdm(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    logger.info(f"Loss at epoch {epoch}: {losses}")


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    # lr = args.lr * (0.1 ** (epoch // 20))
    # for param_group in optimizer.param_groups:
    # param_group['lr'] = lr
    try:
        last_lr = scheduler.get_last_lr()[0]
    except Exception:
        last_lr = optimizer.param_groups[0]["lr"]
    if epoch > args.warmup_epochs and last_lr <= args.min_lr:
        return
    scheduler.step()


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
    logger.info(f"auc {ap}")
    # logger.info(f"roc auc {roc_auc}")
    # logger.info(f"pr auc {pr_auc}")
    return ap


def validate_model(
    val_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn._Loss,
    args: argparse.Namespace,
) -> float:
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.NONE)

    # switch to evaluate mode
    model.eval()

    y_true = []
    y_pred = []
    # total_pred = 0
    # total_target = 0
    pos_list: list[float] = []
    with torch.no_grad():
        end = time.time()
        for images, target in tqdm(val_loader):
            if len(images) == 1:
                continue

            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            # logger.info("Output size: {}".format(output.size()))

            loss = criterion(output, target)

            probability = torch.nn.functional.softmax(output, dim=1)
            probability = np.squeeze(probability.cpu().numpy())
            """
            pos = np.where(target.cpu().numpy()==1)
            if target[pos].sum() > 0:
                logger.info(f"Positives: {np.where(target.cpu().numpy()==1)}")
                logger.info("Pos Probs: {}, {}".format(probability[pos], target[pos]))
            total_pred += probability[pos][:,1].sum()
            pos_list.extend(list(probability[pos][:,1]))
            total_target += target[pos].sum()
            average = total_pred/total_target
            if target[pos].sum() > 0:
                logger.info(
                    "Running average pos score: "
                    f"{total_pred}, {total_target}, {average}"
                )
            """
            try:
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
    logger.info(f"Pred list scores: {sorted(pos_list)}")
    return calculate_performance(y_true, y_pred)


def save_checkpoint(
    state: dict[str, Any],
    filename: os.PathLike[str] | str = "checkpoint.pth",
) -> None:
    torch.save(state, str(filename))


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
