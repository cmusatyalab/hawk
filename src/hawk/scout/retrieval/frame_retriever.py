# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import copy
import os
import time
from pathlib import Path
from typing import Iterable, cast

import cv2
import numpy as np
import numpy.typing as npt
from logzero import logger

from ...classes import NEGATIVE_CLASS
from ...objectid import ObjectId
from ...proto.messages_pb2 import FileDataset
from ..stats import collect_metrics_total
from .retriever import LegacyRetrieverMixin, Retriever


class FrameRetriever(Retriever, LegacyRetrieverMixin):
    def __init__(self, mission_id: str, dataset: FileDataset):
        super().__init__(mission_id)
        self.network = False
        self._dataset = dataset
        self._timeout = dataset.timeout
        self._resize = dataset.resizeTile
        index_file = Path(self._dataset.dataPath)
        self.tilesize = 256 if self._dataset.tileSize == 0 else self._dataset.tileSize
        self.overlap = 100 if 0.5 * self.tilesize > 100 else 0
        self.padding = True
        self.slide = self.tilesize - self.overlap

        self.images = [Path(line) for line in index_file.read_text().splitlines()]

        self.total_tiles = len(self.images)
        self.total_images.set(self.total_tiles)
        self.total_objects.set(self.total_tiles)

    def save_tile(
        self,
        img: npt.NDArray[np.uint8],
        imagename: Path,
        subimgname: str,
        left: int,
        up: int,
    ) -> Path:
        subimg = copy.deepcopy(
            img[up : (up + self.tilesize), left : (left + self.tilesize)]
        )
        outpath = imagename.parent.joinpath(subimgname)
        if self.padding:
            h, w, c = np.shape(subimg)
            outimg = np.zeros((self.tilesize, self.tilesize, c), dtype=np.uint8)
            outimg[0:h, 0:w, :] = subimg
        else:
            outimg = subimg
        cv2.imwrite(os.fspath(outpath), outimg)
        return outpath

    def split_frame(self, frame: Path) -> Iterable[Path]:
        image = cast(npt.NDArray[np.uint8], cv2.imread(str(frame)))

        width = np.shape(image)[1]
        height = np.shape(image)[0]

        left, up = 0, 0
        while left < width:
            if left + self.tilesize >= width:
                left = max(width - self.tilesize, 0)
            up = 0
            while up < height:
                if up + self.tilesize >= height:
                    up = max(height - self.tilesize, 0)
                # right = min(left + self.tilesize, width - 1)
                # down = min(up + self.tilesize, height - 1)
                subimgname = f"{frame.stem}__{left}___{up}{frame.suffix}"
                yield self.save_tile(image, frame, subimgname, left, up)

                if up + self.tilesize >= height:
                    break
                up += self.slide

            if left + self.tilesize >= width:
                break
            left += self.slide

    def stream_objects(self) -> None:
        super().stream_objects()
        assert self._context is not None

        for key in self.images:
            time_start = time.time()
            if self._stop_event.is_set():
                break

            self.retrieved_images.inc()

            self._context.log(f"RETRIEVE: File {key}")
            tiles = list(self.split_frame(key))
            elapsed = time.time() - self._start_time
            logger.info(f"Retrieved Image:{key} Tiles:{len(tiles)} @ {elapsed}")

            # bump total_objects to account for the tiles in this frame
            self.total_objects.inc(len(tiles) - 1)

            for image_path in tiles:
                object_id = ObjectId(f"/{NEGATIVE_CLASS}/collection/id/{image_path}")
                self.put_objectid(object_id)

            retrieved_tiles = collect_metrics_total(self.retrieved_objects)
            logger.info(f"{retrieved_tiles} / {self.total_tiles} RETRIEVED")
            time_passed = time.time() - time_start
            if time_passed < self._timeout:
                time.sleep(self._timeout - time_passed)
