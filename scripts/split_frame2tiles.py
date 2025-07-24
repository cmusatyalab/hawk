# Source:
# https://github.com/CAPTAIN-WHU/DOTA_devkit/blob/master/ImgSplit_multi_process.py

import copy
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml


def split_single_warp(name, split_base, extent) -> None:
    split_base.split_frame(name, extent)


def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    needExtFilter = ext is not None
    for root, _dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if not needExtFilter or extension in ext:
                allfiles.append(filepath)
    return allfiles


def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])


class Frame2TileSplitter:
    def __init__(
        self,
        image_dir,
        label_dir,
        code="utf-8",
        gap=100,
        tilesize=256,
        ext=".png",
        padding=True,
        num_process=8,
    ) -> None:
        self.code = code
        self.gap = gap
        self.tilesize = tilesize
        self.slide = self.tilesize - self.gap
        self.imagepath = image_dir
        output_dir = Path(image_dir)
        output_dir = output_dir.parent / str(output_dir.name) + "tiles"
        self.outimagepath = os.path.join(output_dir, "images")
        self.outlabelpath = os.path.join(output_dir, "labels")
        self.ext = ext
        self.padding = padding
        self.num_process = num_process
        self.pool = Pool(num_process)

        if not os.path.isdir(self.outimagepath):
            os.mkdir(self.outimagepath)
        if not os.path.isdir(self.outlabelpath):
            os.mkdir(self.outlabelpath)

    def save_tile(self, img, subimgname, left, up):
        subimg = copy.deepcopy(
            img[up : (up + self.tilesize), left : (left + self.tilesize)],
        )
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        h, w, c = np.shape(subimg)
        if self.padding:
            outimg = np.zeros((self.tilesize, self.tilesize, 3))
            outimg[0:h, 0:w, :] = subimg
            cv2.imwrite(outdir, outimg)
        else:
            cv2.imwrite(outdir, subimg)

        return outdir

    def split_frame(self, name, extent):
        image = cv2.imread(os.path.join(self.imagepath, name + extent))

        outbasename = name + "__"
        weight = np.shape(image)[1]
        height = np.shape(image)[0]

        left, up = 0, 0
        tiles = []
        while left < weight:
            if left + self.tilesize >= weight:
                left = max(weight - self.tilesize, 0)
            up = 0
            while up < height:
                if up + self.tilesize >= height:
                    up = max(height - self.tilesize, 0)
                # right = min(left + self.tilesize, weight - 1)
                # down = min(up + self.tilesize, height - 1)
                subimgname = outbasename + str(left) + "___" + str(up)
                # self.f_sub.write(f"{name} {subimgname} {left} {up}\n")
                tile = self.save_tile(image, subimgname, left, up)
                tiles.append(tile)
                if up + self.tilesize >= height:
                    break
                up = up + self.slide
            if left + self.tilesize >= weight:
                break
            left = left + self.slide

        return tiles

    def splitdata(self) -> None:
        imagelist = GetFileFromThisRootDir(self.imagepath)
        imagenames = [
            custombasename(x) for x in imagelist if (custombasename(x) != "Thumbs")
        ]

        worker = partial(split_single_warp, split_base=self, extent=self.ext)
        self.pool.map(worker, imagenames)


if __name__ == "__main__":
    config_path = sys.argv[1]  # path to config file
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_config = config["home-params"]
    image_dir = dataset_config["image_dir"]
    label_dir = dataset_config["label_dir"]
    split = Frame2TileSplitter(
        image_dir,
        label_dir,
        gap=200,
        tilesize=1024,
        num_process=8,
    )
    split.splitdata(1)
