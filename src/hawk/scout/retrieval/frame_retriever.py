# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import copy
import io
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import numpy.typing as npt
from logzero import logger
from PIL import Image

from ...proto.messages_pb2 import FileDataset
from ..core.attribute_provider import HawkAttributeProvider
from ..core.object_provider import ObjectProvider
from .retriever import Retriever


class FrameRetriever(Retriever):
    def __init__(self, dataset: FileDataset):
        super().__init__()

        self._dataset = dataset
        self._timeout = dataset.timeout
        self._resize = dataset.resizeTile
        index_file = Path(self._dataset.dataPath)
        self.tilesize = 256 if self._dataset.tileSize == 0 else self._dataset.tileSize
        self.overlap = 100 if 100 < 0.5 * self.tilesize else 0
        self.padding = True
        self.slide = self.tilesize - self.overlap

        self.images = [Path(line) for line in index_file.read_text().splitlines()]
        self._stats.total_objects = len(self.images)
        self._stats.total_images = len(self.images)

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
        cv2.imwrite(outpath, outimg)
        return outpath

    def split_frame(self, frame: Path) -> Iterable[Path]:
        image = cv2.imread(str(frame))

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

        for key in self.images:
            time_start = time.time()
            if self._stop_event.is_set():
                break
            self._stats.retrieved_images += 1
            self._context.log(f"RETRIEVE: File {key}")
            tiles = list(self.split_frame(key))
            logger.info(
                "Retrieved Image:{} Tiles:{} @ {}".format(
                    key, len(tiles), time.time() - self._start_time
                )
            )
            for image_path in tiles:
                tmpfile = io.BytesIO()
                label = 0
                image = Image.open(image_path).convert("RGB")
                image.save(tmpfile, format="JPEG", quality=85)
                content = tmpfile.getvalue()

                object_id = f"/{label}/collection/id/{image_path}"
                attributes = self.set_tile_attributes(object_id, str(label))
                self._stats.retrieved_tiles += 1

                self.result_queue.put_nowait(
                    ObjectProvider(
                        object_id,
                        content,
                        HawkAttributeProvider(attributes, image_path, self._resize),
                        int(label),
                    )
                )
            logger.info(f"{self._stats.retrieved_tiles} / {self.total_tiles} RETRIEVED")
            time_passed = time.time() - time_start
            if time_passed < self._timeout:
                time.sleep(self._timeout - time_passed)
