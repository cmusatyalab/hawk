# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Augments the single input image provided by user to a set of five"""

import os
import shutil
import sys
import uuid
import warnings
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image

warnings.simplefilter("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_SHOT = 5

train_transform = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ColorJitter(),
    ]
)


def AugmentSave(destdir: Path, pil_image: Image.Image) -> None:
    image = np.array(pil_image)
    for _ in range(1, N_SHOT):
        transform = A.Compose([t for t in train_transform])
        image_transform = transform(image=image)["image"]
        transformed_img = Image.fromarray(image_transform)
        transformed_name = destdir.joinpath(str(uuid.uuid4())).with_suffix(".png")
        transformed_img.save(os.fspath(transformed_name))


def main() -> None:
    train_dataset = Path(sys.argv[1])
    image_name = Path(sys.argv[2])
    assert image_name.exists()

    # Target class '0'
    target_class = train_dataset / "0"

    # Create directory for results
    if target_class.exists():
        shutil.rmtree(target_class)
    target_class.mkdir()
    shutil.copy(image_name, target_class)

    image = Image.open(os.fspath(image_name))
    AugmentSave(target_class, image)


if __name__ == "__main__":
    main()
