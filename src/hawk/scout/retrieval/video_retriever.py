# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import copy
import multiprocessing as mp
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import numpy.typing as npt
from logzero import logger

from ...objectid import ObjectId
from .retriever import Retriever, RetrieverConfig
from .retriever_mixins import LegacyRetrieverMixin
from .video_parser import produce_video_frames


class VideoRetrieverConfig(RetrieverConfig):
    video_path: Path
    sampling_rate_fps: int = 1
    width: int = 4000
    height: int = 3000
    tile_size: int = 250
    timeout: float = 8.0


class VideoRetriever(Retriever, LegacyRetrieverMixin):
    config_class = VideoRetrieverConfig
    config: VideoRetrieverConfig

    def __init__(self, config: VideoRetrieverConfig) -> None:
        super().__init__(config)

        self._start_time = time.time()
        self.padding = True
        self.overlap = 100 if 0.5 * self.config.tile_size > 100 else 0
        self.slide = 250
        self.video_file_path = self.config.video_path
        self.frame_producer_queue: mp.Queue[tuple[str, npt.NDArray[np.uint8]]] = (
            mp.Queue(20)
        )
        # self.video_frame_producer = VideoFrameProducer(self.video_file_path)
        p = mp.Process(
            target=produce_video_frames,
            args=(self.frame_producer_queue, self.video_file_path),
            name="Frame Producer",
        )
        p.start()
        # then create new process and start producer function.

        self.temp_image_dir = Path("/srv/diamond/video_stream_temp_image_dir")
        # self.temp_image_dir.mkdir(exist_ok=True)

        # create temp directories on scout to store frames and tiles
        self.temp_tile_dir = Path("/srv/diamond/video_stream_temp_tile_dir")
        self.temp_tile_dir.mkdir(exist_ok=True)

        self.total_images.set(0)
        self.total_objects.set(0)
        # self.total_tiles = 192 * len(os.listdir(self.temp_image_dir))

        # hardcoded for now, but needs to be
        # number of tiles per image x total expected images in video
        self.total_tiles = 192 * 600

    def _save_tile(
        self,
        img: npt.NDArray[np.uint8],
        subimgname: str,
        left: int,
        up: int,
    ) -> Path:
        subimg = copy.deepcopy(
            img[
                up : (up + self.config.tile_size),
                left : (left + self.config.tile_size),
            ],
        )

        tile_name = self.config.video_path.with_suffix(f"{subimgname}.jpeg").name
        tile_path = self.temp_tile_dir.joinpath(tile_name)

        h, w, c = np.shape(subimg)
        outimg = cv2.resize(subimg, (self.config.tile_size, self.config.tile_size))
        logger.info("About to write tile...")
        try:
            cv2.imwrite(str(tile_path), outimg)
        except Exception as e:
            logger.info(e)
        return tile_path

    def _split_frame(
        self,
        frame_name: str,
        frame: npt.NDArray[np.uint8],
    ) -> Iterator[Path]:
        logger.info(self.frame_producer_queue.qsize())
        # frame = cv2.imread(os.path.join(self.temp_image_dir, frame_name))
        basename = self.video_file_path.stem
        outbasename = frame_name.split(".")[0] + "_"
        num_tile_rows = int(self.config.height / self.config.tile_size)
        num_tile_cols = int(self.config.width / self.config.tile_size)
        for row in range(num_tile_rows):
            for col in range(num_tile_cols):
                tile_size = self.config.tile_size
                subimgname = f"{outbasename}{col * tile_size}_{row * tile_size}"
                # tile = self.save_tile(
                #     frame, subimgname, col*tile_size, row*tile_size
                # )
                tile = frame[
                    row * tile_size : (row * tile_size + tile_size),
                    col * tile_size : (col * tile_size + tile_size),
                ]
                # tile = copy.deepcopy(
                #     frame[row * tile_size: (row * tile_size + tile_size),
                #           col * tile_size: (col * tile_size + tile_size)]
                # )

                outdir = self.temp_tile_dir.joinpath(f"{basename}{subimgname}.jpeg")
                # h, w, c = np.shape(subimg)
                resized_tile = cv2.resize(tile, (256, 256))
                cv2.imwrite(str(outdir), resized_tile)
                yield outdir

    def get_next_objectid(self) -> Iterator[ObjectId | None]:
        assert self._context is not None

        logger.info(self.video_file_path)
        frame_count = 1
        num_retrieved_images = 0
        # for frame_name in os.listdir(self.temp_image_dir):

        while True:
            logger.info("Waiting for frame from queue...")
            frame_name, frame = self.frame_producer_queue.get()

            logger.info("Preparing to tile: ")

            self.total_images.inc()
            frame_count += 1
            num_retrieved_images += 1

            tiles = list(self._split_frame(frame_name, frame))

            self.total_objects.inc(len(tiles))

            delta_t = time.time() - self._start_time
            logger.info(
                f"Retrieved Image: Frame # {num_retrieved_images} "
                f"Tiles:{len(tiles)} @ {delta_t}",
            )
            for tile_path in tiles:
                rel_path = tile_path.resolve().relative_to(self.config.data_root)
                yield ObjectId(f"/negative/collection/id/{rel_path}")

            yield None

        self._context.log(f"RETRIEVE: File # {num_retrieved_images}")

    # shutil.rmtree(self.temp_tile_dir)
