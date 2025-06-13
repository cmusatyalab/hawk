# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import copy
import multiprocessing as mp
import time
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from logzero import logger

from ...classes import NEGATIVE_CLASS
from ...objectid import ObjectId
from ...proto.messages_pb2 import Streaming_Video
from ..stats import collect_metrics_total
from .retriever import Retriever
from .retriever_mixins import LegacyRetrieverMixin
from .video_parser import produce_video_frames


class VideoRetriever(Retriever, LegacyRetrieverMixin):
    def __init__(self, mission_id: str, dataset: Streaming_Video):
        super().__init__(mission_id)
        self.network = False
        self._dataset = dataset
        self.timeout = 20
        self._start_time = time.time()
        self.padding = True
        self.tilesize = 250
        self.overlap = 100 if 0.5 * self.tilesize > 100 else 0
        self.slide = 250
        self.video_file_path = dataset.video_path
        self.frame_producer_queue: mp.Queue[Tuple[str, npt.NDArray[np.uint8]]] = (
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
        self.tile_width = self._dataset.tile_width
        self.tile_height = self._dataset.tile_height
        self.video_sampling_rate = self._dataset.sampling_rate_fps
        self.frame_width = self._dataset.width
        self.frame_height = self._dataset.height

        self.temp_image_dir = Path("/srv/diamond/video_stream_temp_image_dir")
        # self.temp_image_dir.mkdir(exist_ok=True)

        # create temp directory on scout to store carved tiles
        self.temp_tile_dir = Path("/srv/diamond/video_stream_temp_tile_dir")
        self.temp_tile_dir.mkdir(exist_ok=True)

        self.total_images.set(0)
        self.total_objects.set(0)
        # self.total_tiles = 192 * len(os.listdir(self.temp_image_dir))

        # hardcoded for now, but needs to be
        # number of tiles per image x total expected images in video
        self.total_tiles = 192 * 600

    def save_tile(
        self, img: npt.NDArray[np.uint8], subimgname: str, left: int, up: int
    ) -> Path:
        subimg = copy.deepcopy(
            img[up : (up + self.tilesize), left : (left + self.tilesize)]
        )

        basename = self.video_file_path.split("/")[-1].split(".")[0]
        outdir = self.temp_tile_dir.joinpath(f"{basename}{subimgname}.jpeg")

        h, w, c = np.shape(subimg)
        outimg = cv2.resize(subimg, (256, 256))
        logger.info("About to write tile...")
        try:
            cv2.imwrite(str(outdir), outimg)
        except Exception as e:
            logger.info(e)
        return outdir

    def split_frame(
        self, frame_name: str, frame: npt.NDArray[np.uint8]
    ) -> Iterable[Path]:
        logger.info(self.frame_producer_queue.qsize())
        # frame = cv2.imread(os.path.join(self.temp_image_dir, frame_name))
        basename = self.video_file_path.split("/")[-1].split(".")[0]
        outbasename = frame_name.split(".")[0] + "_"
        width = 4000
        height = 3000
        num_tile_rows = int(height / self.tilesize)
        num_tile_cols = int(width / self.tilesize)
        for row in range(num_tile_rows):
            for col in range(num_tile_cols):
                subimgname = f"{outbasename}{col * self.tilesize}_{row * self.tilesize}"
                # tile = self.save_tile(
                #     frame, subimgname, col*self.tilesize, row*self.tilesize
                # )
                tile = frame[
                    row * self.tilesize : (row * self.tilesize + self.tilesize),
                    col * self.tilesize : (col * self.tilesize + self.tilesize),
                ]
                # tile = copy.deepcopy(
                #     frame[row * self.tilesize: (row * self.tilesize + self.tilesize),
                #           col * self.tilesize: (col * self.tilesize + self.tilesize)]
                # )

                outdir = self.temp_tile_dir.joinpath(f"{basename}{subimgname}.jpeg")
                # h, w, c = np.shape(subimg)
                resized_tile = cv2.resize(tile, (256, 256))
                cv2.imwrite(str(outdir), resized_tile)
                yield outdir

    def stream_objects(self) -> None:
        # wait for mission context to be added
        super().stream_objects()
        assert self._context is not None

        self._start_time = time.time()
        logger.info(self.video_file_path)
        frame_count = 1
        num_retrieved_images = 0
        # for frame_name in os.listdir(self.temp_image_dir):

        while not self._stop_event.is_set():
            logger.info("Waiting for frame from queue...")
            frame_name, frame = self.frame_producer_queue.get()
            logger.info("Preparing to tile: ")
            if self._stop_event.is_set():
                logger.info("Stop event is set...")
                break

            self.total_images.inc()
            self.retrieved_images.inc()
            frame_count += 1
            num_retrieved_images += 1
            tiles = list(self.split_frame(frame_name, frame))
            self.total_objects.inc(len(tiles))

            delta_t = time.time() - self._start_time
            logger.info(
                f"Retrieved Image: Frame # {num_retrieved_images} "
                f"Tiles:{len(tiles)} @ {delta_t}"
            )
            for tile_path in tiles:
                object_id = ObjectId(f"/{NEGATIVE_CLASS}/collection/id/{tile_path}")
                self.put_objectid(object_id)
            time.sleep(8)

            retrieved_tiles = collect_metrics_total(self.retrieved_objects)
            logger.info(f"{retrieved_tiles} / {self.total_tiles} RETRIEVED")
            # time_passed = time.time() - self._start_time
            # if time_passed < self.timeout:
            #  time.sleep(self.timeout - time_passed)

        self._context.log(f"RETRIEVE: File # {num_retrieved_images}")
        # shutil.rmtree(self.temp_tile_dir)
