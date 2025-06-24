# SPDX-FileCopyrightText: 2022-2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import copy
import os
import time
from pathlib import Path
from typing import Iterator, cast

import cv2
import numpy as np
import numpy.typing as npt
from logzero import logger

from ...objectid import ObjectId
from ..stats import collect_metrics_total
from .retriever import Retriever, RetrieverConfig
from .retriever_mixins import LegacyRetrieverMixin


class FrameRetrieverConfig(RetrieverConfig):
    index_path: Path  # file that contains the index
    timeout: float = 20.0  # seconds per frame (batch of tiles)
    tile_size: int = 256  # desired tile size


class FrameRetriever(Retriever, LegacyRetrieverMixin):
    config_class = FrameRetrieverConfig
    config: FrameRetrieverConfig

    def __init__(self, config: FrameRetrieverConfig) -> None:
        super().__init__(config)

        index_file = (self.config.data_root / self.config.index_path).resolve()

        # make sure index_file is inside the configured data_root.
        # Path.relative_to raises ValueError when it is not a subtree.
        index_file.relative_to(self.config.data_root)

        self.overlap = 100 if 0.5 * self.config.tile_size > 100 else 0
        self.padding = True
        self.slide = self.config.tile_size - self.overlap

        self.images = index_file.read_text().splitlines()
        self.total_tiles = len(self.images)

        self.total_images.set(self.total_tiles)
        self.total_objects.set(self.total_tiles)

    def _save_tile(
        self,
        img: npt.NDArray[np.uint8],
        imagename: Path,
        subimgname: str,
        left: int,
        up: int,
    ) -> Path:
        subimg = copy.deepcopy(
            img[
                up : (up + self.config.tile_size), left : (left + self.config.tile_size)
            ]
        )
        outpath = imagename.parent.joinpath(subimgname)
        if self.padding:
            h, w, c = np.shape(subimg)
            outimg = np.zeros(
                (self.config.tile_size, self.config.tile_size, c), dtype=np.uint8
            )
            outimg[0:h, 0:w, :] = subimg
        else:
            outimg = subimg
        cv2.imwrite(os.fspath(outpath), outimg)
        return outpath

    def _split_frame(self, frame: Path) -> Iterator[Path]:
        image = cast(npt.NDArray[np.uint8], cv2.imread(str(frame)))

        width = np.shape(image)[1]
        height = np.shape(image)[0]

        left, up = 0, 0
        while left < width:
            if left + self.config.tile_size >= width:
                left = max(width - self.config.tile_size, 0)
            up = 0
            while up < height:
                if up + self.config.tile_size >= height:
                    up = max(height - self.config.tile_size, 0)
                # right = min(left + self.config.tile_size, width - 1)
                # down = min(up + self.config.tile_size, height - 1)
                subimgname = f"{frame.stem}__{left}___{up}{frame.suffix}"
                yield self._save_tile(image, frame, subimgname, left, up)

                if up + self.config.tile_size >= height:
                    break
                up += self.slide

            if left + self.config.tile_size >= width:
                break
            left += self.slide

    def get_next_objectid(self) -> Iterator[ObjectId]:
        assert self._context is not None
        for key in self.images:
            time_start = time.time()

            image_path = self.config.data_root.joinpath(key).resolve()
            image_path.relative_to(self.config.data_root)

            self._context.log(f"RETRIEVE: File {image_path}")
            tiles = list(self._split_frame(image_path))

            self.retrieved_images.inc()

            elapsed = time.time() - self._start_time
            logger.info(f"Retrieved Image:{key} Tiles:{len(tiles)} @ {elapsed}")

            # bump total_objects to account for the tiles in this frame
            self.total_objects.inc(len(tiles) - 1)

            for tile_path in tiles:
                rel_path = tile_path.relative_to(self.config.data_root)
                object_id = ObjectId(f"/negative/collection/id/{rel_path}")
                yield object_id

            retrieved_tiles = collect_metrics_total(self.retrieved_objects)
            logger.info(f"{retrieved_tiles} / {self.total_tiles} RETRIEVED")
            time_passed = time.time() - time_start
            if time_passed < self.config.timeout:
                time.sleep(self.config.timeout - time_passed)
