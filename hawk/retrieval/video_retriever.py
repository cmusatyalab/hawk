# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import copy
import gc
import cv2
import io
import queue
import os, shutil, subprocess
import random
import numpy as np 
import time
import threading
from typing import Iterable, Sized
from collections import defaultdict
from PIL import Image
from logzero import logger

from hawk.core.object_provider import ObjectProvider
from hawk.proto.messages_pb2 import HawkObject, FileDataset, Streaming_Video
from hawk.retrieval.diamond_attribute_provider import DiamondAttributeProvider
from hawk.retrieval.retriever import Retriever
from hawk.retrieval.retriever_stats import RetrieverStats
from hawk.core.utils import get_server_ids
from hawk.core.utils import ATTR_GT_LABEL

class VideoRetriever(Retriever):

    def __init__(self, dataset: Streaming_Video):
        super().__init__()
        self._dataset = dataset
        self._start_event = threading.Event()
        self._stop_event = threading.Event()
        self._command_lock = threading.RLock()
        stats_keys = ['total_objects', 'total_images', 'dropped_objects',
                      'false_negatives', 'retrieved_images', 'retrieved_tiles']
        self._stats = {x: 0 for x in stats_keys}
        self.timeout = 20
        self._start_time = time.time()
        self.result_queue = queue.Queue()
        self.padding = True
        self.tilesize = 250
        self.overlap = 100 if 100 < 0.5 * self.tilesize else 0
        self.slide = 250
        self.video_file_path = dataset.video_path
        self.tile_width = self._dataset.tile_width
        self.tile_height = self._dataset.tile_height
        self.video_sampling_rate = self._dataset.sampling_rate_fps
        self.frame_width = self._dataset.width
        self.frame_height = self._dataset.height
        self.temp_image_dir = r"/srv/diamond/video_stream_temp_image_dir"
        self.temp_tile_dir = r"/srv/diamond/video_stream_temp_tile_dir"
        #if not os.path.exists(self.temp_image_dir):
         #   os.mkdir(self.temp_image_dir)
        if not os.path.exists(self.temp_tile_dir):
            os.mkdir(self.temp_tile_dir)  ## create temp directoy on scout to store carved tiles
        ## bash comamnd to mount the fifthcraig folder
        #subprocess.run(["ffmpeg", "-i", self.video_file_path, "-vf", "fps=1", os.path.join(self.temp_image_dir, "%04d.png")])
        mount_dir = r"/srv/diamond/fifthcraig"
        '''
        if len(os.listdir(mount_dir)) != 0:
            unmount_cmd = ["sudo", "fusermount", "-u", "/srv/diamond/fifthcraig"]
            out = subprocess.check_call(unmount_cmd)  ##unmount the AWS bucket
        else:
            mount_cmd_list  = ["sudo", "/usr/bin/s3fs", "hawk-fifthcraig", "/srv/diamond/fifthcraig", "-o", "ro", "-o", "use_path_request_style", "-o", "url=https://storage.cmusatyalab.org"]
            subprocess.run(mount_cmd_list)
        logger.info(out)
        print(out)
        '''
        self._stats['total_objects'] = 0
        self._stats['total_images'] = 0
        self.total_tiles = 192 * len(os.listdir(self.temp_image_dir))
        
    def save_tile(self, img, subimgname, left, up):
        dirname = self.temp_tile_dir
        subimg = copy.deepcopy(img[up: (up + self.tilesize), left: (left + self.tilesize)])
        outdir = os.path.join(dirname, self.video_file_path.split("/")[-1].split(".")[0] + subimgname + ".jpeg")
        h, w, c = np.shape(subimg)
        outimg = cv2.resize(subimg, (256, 256))
        cv2.imwrite(outdir, outimg)
        return outdir

    def split_frame(self, frame_name):
        #with cv2.imread(os.path.join(self.temp_image_dir, frame_name)) as frame:
        frame = cv2.imread(os.path.join(self.temp_image_dir, frame_name))
        outbasename = frame_name.split(".")[0] + '_'
        width = np.shape(frame)[1]
        height = np.shape(frame)[0]
        num_tile_rows = height / self.tilesize
        num_tile_cols = width / self.tilesize
        tiles = []
        for row in range(int(num_tile_rows)):
            for col in range(int(num_tile_cols)):
                subimgname = outbasename + str(col*self.tilesize) + '_' + str(row*self.tilesize)
                tile = self.save_tile(frame, subimgname, col*self.tilesize, row*self.tilesize)
                tiles.append(tile)
        del frame
        gc.collect()
        return tiles


        '''
        left, up = 0, 0
        tiles = []
        while (left < weight):
            if (left + self.tilesize >= weight):
                left = max(weight - self.tilesize, 0)
            up = 0
            while (up < height):
                if (up + self.tilesize >= height):
                    up = max(height - self.tilesize, 0)
                right = min(left + self.tilesize, weight - 1)
                down = min(up + self.tilesize, height - 1)
                subimgname = outbasename + str(left) + '_' + str(up)
                tile = self.save_tile(frame, subimgname, left, up)
                tiles.append(tile)
                if (up + self.tilesize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.tilesize >= weight):
                break
            else:
                left = left + self.slide   
                
        return tiles
    '''
    def stream_objects(self):
        # wait for mission context to be added
        while self._context is None:
            continue

        self._start_time = time.time()
        logger.info(self.video_file_path)
        frame_count = 1
        num_retrieved_images = 0
        for frame_name in os.listdir(self.temp_image_dir):
            logger.info("Preparing to tile: ")
            if self._stop_event.is_set():
                logger.info("Stop event is set...")
                break

            frame_count += 1
            self._stats['retrieved_images'] += 1
            num_retrieved_images += 1
            tiles = self.split_frame(frame_name)

            logger.info("Retrieved Image: Frame # {} Tiles:{} @ {}".format(
                num_retrieved_images, len(tiles), time.time() - self._start_time))
            for tile_path in tiles:
                content = io.BytesIO()
                label = 0
                image = Image.open(tile_path).convert('RGB')
                image.save(content, format='JPEG', quality=85)
                content = content.getvalue()

                object_id = tile_path
                attributes = {
                    'Device-Name': str.encode(get_server_ids()[0]),
                    '_ObjectID': str.encode(object_id),
                    ATTR_GT_LABEL: str.encode(str(label)),
                }
                self._stats['retrieved_tiles'] += 1

                self.result_queue.put_nowait(
                    ObjectProvider(object_id, content,
                                   DiamondAttributeProvider(attributes, tile_path, resize=False),
                                   int(label)))
            logger.info("{} / {} RETRIEVED".format(self._stats['retrieved_tiles'], self.total_tiles))
            time_passed = (time.time() - self._start_time)
            # if time_passed < self.timeout:
            #  time.sleep(self.timeout - time_passed)


        self._stats['retrieved_images'] += 1
        if self._context.enable_logfile:
            self._context.log_file.write("{:.3f} {} RETRIEVE: File # {}\n".format(
                time.time() - self._context.start_time, self._context.host_name, num_retrieved_images))

        #shutil.rmtree(self.temp_tile_dir)

    def is_running(self):
        return not self._stop_event.is_set()

    def start(self) -> None:
        with self._command_lock:
            self._start_time = time.time()

        self._start_event.set()
        threading.Thread(target=self.stream_objects, name='stream').start()

    def stop(self) -> None:
        self._stop_event.set()
        '''
        unmount_cmd = ["fusermount", "-u", "/srv/diamond/fifthcraig"]
        subprocess.run(unmount_cmd) ##unmount the AWS bucket
        ### remove temp_tile_dir
        '''

    def get_objects(self) -> Iterable[ObjectProvider]:
        return self.result_queue.get()

    def get_object(self, object_id: str, attributes: Sized = []) -> HawkObject:
        image_path = object_id
        with open(image_path, 'rb') as f:
            content = f.read()

        # Return object attributes
        dct = {
                'Device-Name': str.encode(get_server_ids()[0]),
                '_ObjectID': str.encode(object_id),
              }

        return HawkObject(objectId=object_id, content=content, attributes=dct)

    def get_stats(self) -> RetrieverStats:
        self._start_event.wait()

        with self._command_lock:
            stats = self._stats.copy()

        return RetrieverStats(stats)
