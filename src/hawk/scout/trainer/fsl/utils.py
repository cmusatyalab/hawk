# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import random
from pathlib import Path

import torch
import torch.nn as nn
from logzero import logger
from PIL import Image
from torch.utils.data import Dataset


class TripletData(Dataset):
    def __init__(self, path: Path, transforms, split: str = "train"):
        self.path = path
        self.split = split  # train or valid
        self.classes = [p.name for p in path.glob("*")]
        self.cats = len(self.classes)  # number of categories
        self.transforms = transforms
        self.total_images = len(list(path.glob("*/*")))
        logger.info(f"Num cats {self.cats} Total {self.total_images}")

    def __getitem__(self, idx):
        # our positive class for the triplet
        idx = int(idx % self.cats)
        classname = self.classes[idx]
        # choosing our pair of positive images (im1, im2)
        positives = list(self.path.joinpath(classname).iterdir())
        im1, im2 = random.sample(positives, 2)

        # choosing a negative class and negative image (im3)
        negative_cats = [x for x in range(self.cats)]
        negative_cats.remove(idx)
        negative_idx = int(random.choice(negative_cats))
        negative_cat = self.classes[negative_idx]
        negatives = list(self.path.joinpath(negative_cat).iterdir())
        im3 = random.choice(negatives)

        im1, im2, im3 = (
            self.path.joinpath(classname, im1),
            self.path.joinpath(classname, im2),
            self.path.joinpath(negative_cat, im3),
        )

        image1 = self.transforms(Image.open(im1))
        image2 = self.transforms(Image.open(im2))
        image3 = self.transforms(Image.open(im3))

        return (image1, image2, image3)

    def __len__(self) -> int:
        return self.total_images


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    # Distances in embedding space is calculated in euclidean
    def forward(self, anchor, positive, negative):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
