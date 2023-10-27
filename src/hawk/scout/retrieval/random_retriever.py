# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import io
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from logzero import logger
from PIL import Image

from ...proto.messages_pb2 import FileDataset
from ..core.attribute_provider import HawkAttributeProvider
from ..core.object_provider import ObjectProvider
from .retriever import Retriever


class RandomRetriever(Retriever):
    def __init__(self, dataset: FileDataset):
        super().__init__()

        self._dataset = dataset
        self._timeout = dataset.timeout
        self._resize = dataset.resizeTile
        logger.info("In RANDOM RETRIEVER INIT...")
        logger.info(f"Resize tile: {self._resize}")

        index_file = self._dataset.dataPath
        self._data_root = Path(index_file).parent.parent
        contents = open(index_file).read().splitlines()
        self.total_tiles = len(contents)

        num_tiles = self._dataset.numTiles
        key_len = math.ceil(self.total_tiles / num_tiles)

        keys = np.arange(key_len)
        per_frame = np.array_split(contents, key_len)

        self.img_tile_map = defaultdict(list)
        for i, tiles_per_frame in enumerate(per_frame):
            k = keys[i]
            for content in tiles_per_frame:
                self.img_tile_map[k].append(content)

        # random.shuffle(keys)
        self.images = keys
        self._stats.total_objects = self.total_tiles
        self._stats.total_images = len(self.images)

    def stream_objects(self) -> None:
        super().stream_objects()

        for key in self.images:
            time_start = time.time()
            if self._stop_event.is_set():
                break
            self._stats.retrieved_images += 1
            tiles = self.img_tile_map[key]
            logger.info(
                "Retrieved Image:{} Tiles:{} @ {}".format(
                    key, len(tiles), time.time() - self._start_time
                )
            )
            for tile in tiles:
                tmpfile = io.BytesIO()
                parts = tile.split()
                if len(parts) == 1:
                    image_path = parts[0]
                    label = "0"
                else:
                    image_path = parts[0]
                    label = parts[1]

                object_id = f"/{label}/collection/id/" + str(image_path)

                image_path = self._data_root / image_path
                image = Image.open(image_path).convert("RGB")
                image.save(tmpfile, format="JPEG", quality=85)
                content = tmpfile.getvalue()

                attributes = self.set_tile_attributes(object_id, label)
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
