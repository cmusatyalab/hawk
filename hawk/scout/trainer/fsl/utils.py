# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob 
import random 
import numpy as np
from PIL import Image 
from pathlib import Path
from enum import Enum
from torch_snippets import *

class TripletData(Dataset):
    def __init__(self, path, transforms, split="train"):
        self.path = path
        self.split = split    # train or valid
        self.classes = glob.glob(os.path.join(path, '*'))
        self.cats = len(self.classes)       # number of categories
        self.transforms = transforms
        self.total_images = len(glob.glob(os.path.join(path, '*/*')))
        logger.info(f"Num cats {self.cats} Total {self.total_images}")
    
    def __getitem__(self, idx):
        # our positive class for the triplet
        idx = int(idx%self.cats)
        classname = self.classes[idx]
        # choosing our pair of positive images (im1, im2)
        positives = os.listdir(os.path.join(self.path, classname))
        im1, im2 = random.sample(positives, 2)
        
        # choosing a negative class and negative image (im3)
        negative_cats = [x for x in range(self.cats)]
        negative_cats.remove(idx)
        negative_idx = int(random.choice(negative_cats))
        negative_cat = self.classes[negative_idx]
        negatives = os.listdir(os.path.join(self.path, negative_cat))
        im3 = random.choice(negatives)
        
        im1,im2,im3 = os.path.join(self.path, classname, im1), os.path.join(self.path, classname, im2), os.path.join(self.path, negative_cat, im3)
        
        im1 = self.transforms(Image.open(im1))
        im2 = self.transforms(Image.open(im2))
        im3 = self.transforms(Image.open(im3))
        
        return [im1, im2, im3]
        
    def __len__(self):
        return self.total_images


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    # Distances in embedding space is calculated in euclidean
    def forward(self, anchor, positive, negative):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()