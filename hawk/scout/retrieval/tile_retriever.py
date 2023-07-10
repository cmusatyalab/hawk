# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import io
import os
import queue
import random
import threading
import time
from collections import defaultdict
from typing import Iterable, Sized

from logzero import logger
from PIL import Image

from ...proto.messages_pb2 import FileDataset, HawkObject
from ..core.attribute_provider import HawkAttributeProvider
from ..core.object_provider import ObjectProvider
from ..core.utils import get_server_ids
from .retriever import KEYS, Retriever, RetrieverStats


class TileRetriever(Retriever):

    def __init__(self, dataset: FileDataset):
        super().__init__()

        self._dataset = dataset
        self._timeout = dataset.timeout
        self._resize = dataset.resizeTile
        index_file = self._dataset.dataPath
        contents = open(index_file).read().splitlines()
        self.img_tile_map = defaultdict(list)

        self.images = []
        for content in contents:
            # content = "<path> <label>"
            path, label = content.split()
            key = os.path.basename(path).split('_')[0]
            if key not in self.img_tile_map:
                self.images.append(key)
            self.img_tile_map[key].append(content)

        self.total_tiles = len(contents)
        self._stats['total_objects'] = self.total_tiles
        self._stats['total_images'] = len(self.images)


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
            tiles = self.img_tile_map[key]
            logger.info("Retrieved Image:{} Tiles:{} @ {}".format(
                key, len(tiles), time.time() - self._start_time))
            for tile in tiles:
                content = io.BytesIO()
                image_path, label = tile.split()
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
