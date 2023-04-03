# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

"""Augments the single input image provided by user to a set of five
"""

import torch
import torch.nn as nn
import uuid
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

import io
import time
import os
import copy
from glob import glob
from tqdm import tqdm
import warnings
import sys
import shutil

import albumentations as A
from PIL import Image
import cv2
from albumentations.pytorch import ToTensorV2
warnings.simplefilter('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_SHOT = 5
train_dataset = sys.argv[1]
image_name = sys.argv[2]
assert os.path.exists(image_name)

# Target class '0'
# Create directory for results
if os.path.exists(f'{train_dataset}/0'):
    shutil.rmtree(f'{train_dataset}/0')
os.mkdir(f'{train_dataset}/0')
shutil.copy(image_name, f'{train_dataset}/0')

train_transform = A.Compose(
    [   A.Resize(height=256, width=256),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ColorJitter(),
    ]
)

def AugmentSave(image):
    for _ in range(1, N_SHOT):
        transform = A.Compose([t for t in train_transform])
        image_transform = transform(image=image)["image"]
        image_transform = Image.fromarray(image_transform)
        image_transform.save(f'{train_dataset}/0/{str(uuid.uuid4())}.png')

image = np.array(Image.open(image_name))
AugmentSave(image)

