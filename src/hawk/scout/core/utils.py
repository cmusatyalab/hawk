# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import hashlib
import socket
import struct
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Tuple, TypeVar, cast

import numpy as np
import numpy.typing as npt
from logzero import logger
from PIL import Image
from torch.utils.data import Dataset


class StringAttributeCodec:
    """Codec for a null-terminated string."""

    def encode(self, item: str) -> bytes:
        assert isinstance(item, str)
        return str.encode(item + "\0")

    def decode(self, data: bytes) -> str:
        decoded = data.decode()
        if decoded[-1] != "\0":
            raise ValueError(f"Attribute value is not null-terminated: {decoded!s}")
        return decoded[:-1]


class IntegerAttributeCodec:
    """Codec for a 32-bit native-endian integer."""

    def encode(self, item: int) -> bytes:
        assert isinstance(item, int)
        return struct.pack("i", item)

    def decode(self, data: bytes) -> int:
        return cast(int, struct.unpack("i", data)[0])


class AverageMeter:
    """Computes and stores the average and current value
    Imported from
        https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self) -> None:
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


class Dict2Obj:
    """
    Turns a dictionary into a class
    """

    def __init__(self, dictionary: dict[str, Any]) -> None:
        for key in dictionary:
            setattr(self, key, dictionary[key])


T = TypeVar("T")


class BaseImageFromList(Dataset[T]):  # type: ignore[misc]
    """Load dataset from path list"""

    def __init__(
        self,
        image_list: list[Path],
        transform: Callable[[Image.Image], Image.Image],
        label_list: list[int] | None = None,
        limit: int | None = None,
        loader: Callable[[Path], Image.Image] | None = None,
    ):
        self.loader = loader if loader is not None else self.image_loader

        self.transform = transform

        def target_transform(x: Path) -> int:
            # add new code here to convert classes from "car", "truck" to 0,1,2, etc.
            # logger.info("X here: {}".format(x))
            return 0 if str(x).startswith("/0/") else 1

        labels = [target_transform(path) for path in image_list]
        if label_list is None:
            label_list = labels
        self.classes = sorted(set(label_list))
        logger.info(f"Classes: {self.classes}")

        if limit is None:
            limit = len(labels)

        max_count = {k: limit for k in set(label_list)}
        num_count = {k: 0 for k in max_count}
        self.targets = []
        self.imlist = []

        for target, image in zip(label_list, image_list):
            num_count[target] += 1
            if num_count[target] > max_count[target]:
                continue
            self.targets.append(target)
            self.imlist.append(image)

        logger.info(
            f"Number of Dataset(Limit {limit}): \n"
            f" Targets {len(self.targets)} \n"
            f" Positives {sum(self.targets)}"
            f" Labels {set(self.targets)}"
        )

    def image_loader(self, path: Path) -> Image.Image:
        try:
            if path.suffix == ".npy":
                array: npt.NDArray[Any] = np.load(path)
                array = (array - np.min(array)) / (np.max(array) - np.min(array))
                image = Image.fromarray((array * 255).astype(np.uint8))
            else:
                image = Image.open(path).convert("RGB")
        except Exception as e:
            logger.error(e)
            logger.error(path)
            image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3)), "RGB")
        return image

    def __len__(self) -> int:
        return len(self.imlist)


class ImageFromList(BaseImageFromList[Tuple[Image.Image, int]]):
    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        impath = self.imlist[idx]
        target = self.targets[idx]
        img = self.loader(impath)
        img = self.transform(img)
        return img, target


class ImageWithPath(BaseImageFromList[Tuple[Path, Image.Image, int]]):
    """Returns image path with data"""

    def __init__(
        self,
        image_list: list[Path],
        transform: Callable[[Image.Image], Image.Image],
        label_list: list[int] | None = None,
    ):
        self.img_paths = image_list
        super().__init__(image_list, transform, label_list)

    def __getitem__(self, idx: int) -> tuple[Path, Image.Image, int]:
        path = self.img_paths[idx]
        impath = self.imlist[idx]
        img = self.loader(impath)
        img = self.transform(img)
        label = self.targets[idx]
        return path, img, label


def get_server_ids() -> list[str]:
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


def get_example_key(content: bytes, extension: str = ".jpg") -> str:
    return (
        hashlib.sha1(content).hexdigest() + extension
    )  # Will need to modify this in order to save the .npy file to scout dir


def get_weights(targets: list[int], num_classes: int = 2) -> list[float]:
    class_weights = [0.0] * num_classes
    classes, counts = np.unique(targets, return_counts=True)
    for class_id, count in zip(classes, counts):
        class_weights[class_id] = len(targets) / float(count)

    logger.info(f"Class weights: {class_weights}")

    weight = [0.0] * len(targets)
    for idx, val in enumerate(targets):
        weight[idx] = class_weights[val]

    return weight


def log_exceptions(func: Callable[..., T]) -> Callable[..., T]:
    @wraps(func)
    def func_wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            raise e

    return cast(Callable[..., T], func_wrapper)
