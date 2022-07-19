# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import copy
import cv2
import io
import queue
import os
import random
import numpy as np 
import time
import threading
from typing import Iterable, Sized
from collections import defaultdict
from PIL import Image
from logzero import logger

from hawk.core.object_provider import ObjectProvider
from hawk.proto.messages_pb2 import HawkObject, FileDataset
from hawk.retrieval.diamond_attribute_provider import DiamondAttributeProvider
from hawk.retrieval.retriever import Retriever
from hawk.retrieval.retriever_stats import RetrieverStats
from hawk.core.utils import get_server_ids
from hawk.core.utils import ATTR_GT_LABEL

class FileSystemRetriever(Retriever):

    def __init__(self, dataset: FileDataset):
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
        index_file = self._dataset.dataPath
        self.tilesize = 256 if self._dataset.tileSize == 0 else self._dataset.tileSize
        self.overlap = 100 if 100 < 0.5*self.tilesize else 0
        logger.info("Started ret {}".format(index_file))
        contents = open(index_file).read().splitlines()
        self.img_tile_map = defaultdict(list)
        self.padding = True
        self.slide = self.tilesize - self.overlap
        self.images = []
        for content in contents:
            # content = "<path>"
            key = os.path.basename(content).split('_')[0]
            self.images.append(key)

        self._stats['total_objects'] = len(self.images)
        self._stats['total_images'] = len(self.images)
        
    def save_tile(self, img, imagename, subimgname, left, up):
        dirname = os.path.dirname(imagename)
        ext = os.path.basename(imagename).split['.'][-1]
        subimg = copy.deepcopy(img[up: (up + self.tilesize), left: (left + self.tilesize)])
        outdir = os.path.join(dirname, subimgname + ext)
        h, w, c = np.shape(subimg)
        if (self.padding):
            outimg = np.zeros((self.tilesize, self.tilesize, 3))
            outimg[0:h, 0:w, :] = subimg
            cv2.imwrite(outdir, outimg)
        else:
            cv2.imwrite(outdir, subimg)
        
        return outdir

    def split_frame(self, frame):
        name = os.path.basename(frame)
       
        image = cv2.imread(frame)
         
        outbasename = name + '__' 
        weight = np.shape(image)[1]
        height = np.shape(image)[0]

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
                subimgname = outbasename + str(left) + '___' + str(up)
                # self.f_sub.write(name + ' ' + subimgname + ' ' + str(left) + ' ' + str(up) + '\n')
                tile = self.save_tile(image, frame, subimgname, left, up)
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

    def stream_objects(self):

        # wait for mission context to be added
        while self._context is None:
            continue

        self._start_time = time.time()
        for key in self.images:
            time_start = time.time()
            if self._stop_event.is_set():
                break
            self._stats['retrieved_images'] += 1
            if self._context.enable_logfile:
                self._context.log_file.write("{:.3f} {} RETRIEVE: File {}\n".format(
                    time.time() - self._context.start_time, self._context.host_name, key))
            tiles = self.split_frame(key)
            logger.info("Retrieved Image:{} Tiles:{} @ {}".format(
                key, len(tiles), time.time() - self._start_time))
            for image_path in tiles:
                content = io.BytesIO()
                label = 0
                image = Image.open(image_path).convert('RGB')
                image.save(content, format='JPEG', quality=85)
                content = content.getvalue()

                object_id = "/{}/collection/id/".format(label)+image_path
                attributes = {
                    'Device-Name': str.encode(get_server_ids()[0]),
                    '_ObjectID': str.encode(object_id),
                    ATTR_GT_LABEL: str.encode(label),
                }
                self._stats['retrieved_tiles'] += 1

                self.result_queue.put_nowait(
                    ObjectProvider(object_id, content,
                                   DiamondAttributeProvider(attributes, image_path, resize=False),
                                   int(label)))
            logger.info("{} / {} RETRIEVED".format(self._stats['retrieved_tiles'], self.total_tiles))
            time_passed = (time.time() - time_start)
            if time_passed < self.timeout:
                time.sleep(self.timeout - time_passed)

    def is_running(self):
        return not self._stop_event.is_set()

    def start(self) -> None:
        with self._command_lock:
            self._start_time = time.time()

        self._start_event.set()
        threading.Thread(target=self.stream_objects, name='stream').start()

    def stop(self) -> None:
        self._stop_event.set()

    def get_objects(self) -> Iterable[ObjectProvider]:
        return self.result_queue.get()

    def get_object(self, object_id: str, attributes: Sized = []) -> HawkObject:
        image_path = object_id.split("collection/id/")[-1]
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
