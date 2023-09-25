# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import copy
import io
import multiprocessing as mp
import os
import time

import cv2
import numpy as np
from logzero import logger
from PIL import Image

from ...proto.messages_pb2 import Streaming_Video
from ..core.attribute_provider import HawkAttributeProvider
from ..core.object_provider import ObjectProvider
from .retriever import Retriever
from .video_parser import produce_video_frames


class VideoRetriever(Retriever):
    def __init__(self, dataset: Streaming_Video):
        super().__init__()
        self._dataset = dataset
        self.timeout = 20
        self._start_time = time.time()
        self.padding = True
        self.tilesize = 250
        self.overlap = 100 if 100 < 0.5 * self.tilesize else 0
        self.slide = 250
        self.video_file_path = dataset.video_path
        self.frame_producer_queue = mp.Queue(20)
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
        self.temp_image_dir = r"/srv/diamond/video_stream_temp_image_dir"
        self.temp_tile_dir = r"/srv/diamond/video_stream_temp_tile_dir"
        # if not os.path.exists(self.temp_image_dir):
        #   os.mkdir(self.temp_image_dir)
        if not os.path.exists(self.temp_tile_dir):
            # create temp directory on scout to store carved tiles
            os.mkdir(self.temp_tile_dir)

        self._stats["total_objects"] = 1
        self._stats["total_images"] = 1
        # self.total_tiles = 192 * len(os.listdir(self.temp_image_dir))

        # hardcoded for now, but needs to be
        # number of tiles per image x total expected images in video
        self.total_tiles = 192 * 600

    def save_tile(self, img, subimgname, left, up):
        dirname = self.temp_tile_dir
        subimg = copy.deepcopy(
            img[up : (up + self.tilesize), left : (left + self.tilesize)]
        )
        outdir = os.path.join(
            dirname,
            self.video_file_path.split("/")[-1].split(".")[0] + subimgname + ".jpeg",
        )
        h, w, c = np.shape(subimg)
        outimg = cv2.resize(subimg, (256, 256))
        logger.info("About to write tile...")
        try:
            cv2.imwrite(outdir, outimg)
        except Exception as e:
            logger.info(e)
        return outdir

    def split_frame(self, frame_name, frame):
        logger.info(self.frame_producer_queue.qsize())
        # frame = cv2.imread(os.path.join(self.temp_image_dir, frame_name))
        outbasename = frame_name.split(".")[0] + "_"
        width = 4000
        height = 3000
        num_tile_rows = height / self.tilesize
        num_tile_cols = width / self.tilesize
        tiles = []
        dirname = self.temp_tile_dir
        for row in range(int(num_tile_rows)):
            for col in range(int(num_tile_cols)):
                subimgname = (
                    outbasename
                    + str(col * self.tilesize)
                    + "_"
                    + str(row * self.tilesize)
                )
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
                outdir = os.path.join(
                    dirname,
                    self.video_file_path.split("/")[-1].split(".")[0]
                    + subimgname
                    + ".jpeg",
                )
                # h, w, c = np.shape(subimg)
                tile = cv2.resize(tile, (256, 256))
                cv2.imwrite(outdir, tile)
                tiles.append(outdir)
        return tiles

    def stream_objects(self):
        # wait for mission context to be added
        while self._context is None:
            continue

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

            frame_count += 1
            self._stats["retrieved_images"] += 1
            num_retrieved_images += 1
            tiles = self.split_frame(frame_name, frame)

            logger.info(
                "Retrieved Image: Frame # {} Tiles:{} @ {}".format(
                    num_retrieved_images, len(tiles), time.time() - self._start_time
                )
            )
            for tile_path in tiles:
                content = io.BytesIO()
                label = str(0)
                image = Image.open(tile_path).convert("RGB")
                image.save(content, format="JPEG", quality=85)
                content = content.getvalue()

                object_id = tile_path
                """attributes = {
                    'Device-Name': str.encode(get_server_ids()[0]),
                    '_ObjectID': str.encode(object_id),
                    ATTR_GT_LABEL: str.encode(str(label)),
                }"""
                attributes = self.set_tile_attributes(object_id, label)
                self._stats["retrieved_tiles"] += 1

                self.result_queue.put_nowait(
                    ObjectProvider(
                        object_id,
                        content,
                        HawkAttributeProvider(attributes, tile_path, resize=False),
                        int(label),
                    )
                )
            time.sleep(8)
            logger.info(
                "{} / {} RETRIEVED".format(
                    self._stats["retrieved_tiles"], self.total_tiles
                )
            )
            # time_passed = time.time() - self._start_time
            # if time_passed < self.timeout:
            #  time.sleep(self.timeout - time_passed)

        self._stats["retrieved_images"] += 1
        if self._context.enable_logfile:
            self._context.log_file.write(
                "{:.3f} {} RETRIEVE: File # {}\n".format(
                    time.time() - self._context.start_time,
                    self._context.host_name,
                    num_retrieved_images,
                )
            )

        # shutil.rmtree(self.temp_tile_dir)
