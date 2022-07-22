# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only
# Source: https://github.com/pytorch/examples/blob/main/imagenet/main.py
import argparse
import os
import random
import time
import warnings
import numpy as np
from enum import Enum
from logzero import logger
from sklearn.metrics import average_precision_score
from hawk.core.utils import ImageFromList
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--trainpath', type=str, required=True,
                    help='path to tain file')
parser.add_argument('--valpath', type=str, default='',
                    help='path to tain file')
parser.add_argument('--savepath', type=str, required=True,
                    help='path to save trained model')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--num-classes', default=2, type=int, 
                    help='number of total epochs to run')
parser.add_argument('--num-unfreeze', default=0, type=int, 
                    help='number of total epochs to run')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup-epochs', default=5, type=int,
                    help='intial number of epochs for warmup')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--break-epoch', default=-1, type=int,
                    help='break epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min-lr', default=1e-4, type=float,
                    help='minimum learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    start_time = time.time()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=True)
    model, input_size = initialize_model(args.arch, args.num_classes, args.num_unfreeze)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        print('using GPU')
        model = model.cuda()


    cudnn.benchmark = True

    # Data loading code
    train_path = args.trainpath
    train_list = []
    train_labels = []
    with open(train_path, "r") as f:
        contents = f.read().splitlines()
        for content in contents:
            path, label = content.split()
            train_list.append(path)
            train_labels.append(int(label))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = ImageFromList(
        train_list,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        label_list=train_labels,
        limit=500*sum(train_labels))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    args.validate = True if args.valpath else False
    logger.info("Validate {}".format(args.validate))
    if args.validate:
        val_path = args.valpath
        logger.info("Test path {}".format(val_path))
        val_list = []
        val_labels = []

        with open(val_path, "r") as f:
            contents = f.read().splitlines()
            for content in contents:
                path, label = content.split()
                val_list.append(path)
                val_labels.append(int(label))

        val_dataset = ImageFromList(
            val_list,
            transforms.Compose([
                transforms.Resize(input_size + 32),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ]),
            label_list=val_labels,
            limit=500*sum(val_labels))

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    targets = torch.LongTensor(train_dataset.targets)
    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
    total_samples = sum(class_sample_count)
    class_weights = [1 - (float(x) / float(total_samples)) for x in class_sample_count]
    logger.info("Total samples {} Class Weight {}".format(total_samples, class_weights))
    criterion = nn.CrossEntropyLoss(weight = torch.Tensor(class_weights), label_smoothing=0.1).cuda()

    # define loss function (criterion), optimizer, and learning rate scheduler
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = StepLR(optimizer, step_size=100, gamma=0.9)
    lr_warmup_epochs = args.warmup_epochs
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.5, total_iters=lr_warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_lr_scheduler, lr_scheduler], milestones=[lr_warmup_epochs])


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    epoch_count = 0
    args.break_epoch = args.epochs if args.break_epoch == -1 else args.break_epoch

    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_seed()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        logger.info("Epoch {}".format(epoch))
        if args.validate:
            # evaluate on validation set
            acc1 = validate_model(val_loader, model, criterion, args)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                logger.info("Saving model AUC: {}".format(best_acc1))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict()
                }, args.savepath)

        adjust_learning_rate(optimizer, scheduler, epoch, args)
        epoch_count += 1
        if epoch_count >= args.break_epoch:
            if not args.validate:
                logger.info("Saving last model")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict()
                }, args.savepath)

            break

    end_time = time.time()
    print(end_time - start_time)

    # EMA: Averaging models
    if args.validate:
        best_checkpoint = torch.load(args.savepath)
        curr_model = best_checkpoint['state_dict']
    else:
        curr_model = model.state_dict()

    if args.resume:
        # curr_model = curr_model.detach().cpu()
        checkpoint = torch.load(args.resume) 
        old_model = checkpoint['state_dict']

        for key in old_model:
            curr_model[key] = (curr_model[key] + old_model[key]) / 2.

    model.load_state_dict(curr_model)

    if args.validate:
        best_auc = validate_model(val_loader, model, criterion, args)
        logger.info("Best TEST AUC {}".format(best_auc))

    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()
    }, args.savepath)


def set_parameter_requires_grad(model, unfreeze=0):
    len_layers = len(list(model.children()))
    num_freeze = len_layers - unfreeze
    if unfreeze == -1:
        num_freeze = 0

    count = 0
    for child in model.children():
        count += 1
        if count < num_freeze:
            for param in child.parameters():
                param.requires_grad = False


def initialize_model(arch, num_classes, unfreeze=0):
    model_ft = None
    input_size = 0
    model_ft = models.__dict__[arch](pretrained=True)

    if "resnet" in arch:
        """ Resnet
        """
        set_parameter_requires_grad(model_ft, unfreeze)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif "alexnet" in arch:
        """ Alexnet
        """
        set_parameter_requires_grad(model_ft, unfreeze)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif "vgg" in arch:
        """ VGG11_bn
        """
        set_parameter_requires_grad(model_ft, unfreeze)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif "squeezenet" in arch:
        """ Squeezenet
        """
        set_parameter_requires_grad(model_ft, unfreeze)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif "densenet" in arch:
        """ Densenet
        """
        set_parameter_requires_grad(model_ft, unfreeze)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif "inception" in arch:
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        set_parameter_requires_grad(model_ft, unfreeze)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    elif "efficientnet" in arch:
        set_parameter_requires_grad(model_ft, unfreeze)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
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

def adjust_learning_rate(optimizer, scheduler, epoch, args):
    # lr = args.lr * (0.1 ** (epoch // 20))
    # for param_group in optimizer.param_groups:
        # param_group['lr'] = lr
    try:
        last_lr = scheduler.get_last_lr()[0]
    except: 
        last_lr = optimizer.param_groups[0]['lr']
    if epoch > args.warmup_epochs and last_lr <= args.min_lr:
        return 
    scheduler.step()
    

def calculate_performance(y_true, y_pred):
    ap = average_precision_score(y_true, y_pred, average=None)
    logger.info("AUC {}".format(ap))
    return ap


def validate_model(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)

    # switch to evaluate mode
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)

            loss = criterion(output, target)

            probability = torch.nn.functional.softmax(output, dim=1)
            probability = np.squeeze(probability.cpu().numpy())
            try:
                probability = probability[:, 1]
                y_pred.extend(probability)
                y_true.extend(target.cpu())
            except:
                probability = probability[1]
                y_pred.append(probability)
                y_true.append(target.cpu()[0])

            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    auc = calculate_performance(y_true, y_pred)

    return auc


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


if __name__ == '__main__':
    main()