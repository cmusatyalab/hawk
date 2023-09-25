# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import hashlib
import socket
import struct
from functools import wraps
from typing import List

import numpy as np
import torch
from logzero import logger
from PIL import Image


class StringAttributeCodec:
    """Codec for a null-terminated string."""

    def encode(self, item):
        assert isinstance(item, str)
        return str.encode(item + "\0")

    def decode(self, data):
        data = data.decode()
        if data[-1] != "\0":
            raise ValueError(f"Attribute value is not null-terminated: {str(data)}")
        return data[:-1]


class IntegerAttributeCodec:
    """Codec for a 32-bit native-endian integer."""

    def encode(self, item):
        assert isinstance(item, int)
        return struct.pack("i", item)

    def decode(self, data):
        return struct.unpack("i", data)[0]


class AverageMeter:
    """Computes and stores the average and current value
    Imported from
        https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
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


class Dict2Obj:
    """
    Turns a dictionary into a class
    """

    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])


class ImageFromList(torch.utils.data.Dataset):
    """Load dataset from path list"""

    def __init__(self, image_list, transform, label_list=None, limit=None, loader=None):
        if loader is None:
            self.loader = self.image_loader
        else:
            self.loader = loader

        self.transform = transform

        def target_transform(x: str) -> int:
            return 1 if "/1/" in x else 0

        labels = [target_transform(path) for path in image_list]
        self.classes = sorted(set(labels))
        if label_list is None:
            label_list = labels

        if limit is None:
            limit = len(labels)

        max_count = {k: limit for k in set(label_list)}
        num_count = {k: 0 for k in max_count}
        self.targets = []
        self.imlist = []

        for target, img in zip(label_list, image_list):
            num_count[target] += 1
            if num_count[target] > max_count[target]:
                continue
            self.targets.append(target)
            self.imlist.append(img)

        logger.info(
            f"Number of Dataset(Limit {limit}): \n"
            f" Targets {len(self.targets)} \n"
            f" Positives {sum(self.targets)}"
            f" Labels {set(self.targets)}"
        )

    def image_loader(self, path):
        assert isinstance(path, str), f"Loader error {path}"
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            logger.error(e)
            logger.error(path)
            image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3)), "RGB")

        return image

    def __getitem__(self, idx):
        impath = self.imlist[idx]
        target = self.targets[idx]
        img = self.loader(impath)
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imlist)


class ImageWithPath(ImageFromList):
    """Returns image path with data"""

    def __init__(self, image_list, transform, label_list=None):
        self.img_paths = image_list
        super().__init__(image_list, transform, label_list)

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        path = self.img_paths[idx]
        return path, img, label


def get_server_ids():
    names = set()
    hostname = socket.getfqdn()
    try:
        for info in socket.getaddrinfo(hostname, None):
            try:
                name = socket.getnameinfo(info[4], socket.NI_NAMEREQD)[0]
                names.add(name)
            except socket.gaierror:
                pass
    except socket.gaierror:
        pass

    return list(names)


def get_example_key(content) -> str:
    return hashlib.sha1(content).hexdigest() + ".jpg"


def get_weights(targets: List[int], num_classes=2) -> List[int]:
    class_weights = [0] * num_classes
    classes, counts = np.unique(targets, return_counts=True)
    for class_id, count in zip(classes, counts):
        class_weights[class_id] = len(targets) / float(count)

    logger.info(f"Class weights: {class_weights}")

    weight = [0] * len(targets)
    for idx, val in enumerate(targets):
        weight[idx] = class_weights[val]

    return weight


def log_exceptions(func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            raise e

    return func_wrapper
