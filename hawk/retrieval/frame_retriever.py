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
from hawk.core.attribute_provider import HawkAttributeProvider
from hawk.retrieval.retriever import Retriever, RetrieverStats, KEYS
from hawk.core.utils import get_server_ids

class FrameRetriever(Retriever):

    def __init__(self, dataset: FileDataset):
        super().__init__()

        self._dataset = dataset
        self._timeout = dataset.timeout
        self._resize = dataset.resizeTile
        index_file = self._dataset.dataPath
        self.tilesize = 256 if self._dataset.tileSize == 0 else self._dataset.tileSize
        self.overlap = 100 if 100 < 0.5*self.tilesize else 0
        contents = open(index_file).read().splitlines()
        self.img_tile_map = defaultdict(list)
        self.padding = True
        self.slide = self.tilesize - self.overlap
        self.images = []
        for content in contents:
            self.images.append(content)

        self._stats['total_objects'] = len(self.images)
        self._stats['total_images'] = len(self.images)
        
    def save_tile(self, img, imagename, subimgname, left, up):
        dirname = os.path.dirname(imagename)
        subimg = copy.deepcopy(img[up: (up + self.tilesize), left: (left + self.tilesize)])
        outdir = os.path.join(dirname, subimgname)
        h, w, c = np.shape(subimg)
        if (self.padding):
            outimg = np.zeros((self.tilesize, self.tilesize, 3))
            outimg[0:h, 0:w, :] = subimg
            cv2.imwrite(outdir, outimg)
        else:
            cv2.imwrite(outdir, subimg)
        
        return outdir

    def split_frame(self, frame):
        name, ext = os.path.basename(frame).split('.')
       
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
                subimgname = outbasename + str(left) + '___' + str(up) + '.' + ext
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
        super().stream_objects()

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
                attributes = self.set_tile_attributes(object_id, label)
                self._stats['retrieved_tiles'] += 1

                self.result_queue.put_nowait(
                    ObjectProvider(object_id, content,
                                   HawkAttributeProvider(attributes, image_path, self._resize),
                                   int(label)))
            logger.info("{} / {} RETRIEVED".format(self._stats['retrieved_tiles'], self.total_tiles))
            time_passed = (time.time() - time_start)
            if time_passed < self._timeout:
                time.sleep(self._timeout - time_passed)