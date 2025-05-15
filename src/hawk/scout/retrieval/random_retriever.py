# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import io
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from logzero import logger
from PIL import Image

from ...classes import NEGATIVE_CLASS, ClassLabel, ClassName
from ...objectid import ObjectId
from ...proto.messages_pb2 import FileDataset
from ..core.attribute_provider import HawkAttributeProvider
from ..core.object_provider import ObjectProvider
from ..stats import collect_metrics_total
from .retriever import Retriever


class RandomRetriever(Retriever):
    def __init__(self, mission_id: str, dataset: FileDataset):
        super().__init__(mission_id)
        self.network = False
        self._dataset = dataset
        self._timeout = dataset.timeout
        self._resize = dataset.resizeTile
        logger.info("In RANDOM RETRIEVER INIT...")
        logger.info(f"Resize tile: {self._resize}")

        index_file = Path(self._dataset.dataPath)
        self._data_root = index_file.parent.parent
        contents = index_file.read_text().splitlines()
        self.total_tiles = len(contents)

        num_tiles = self._dataset.numTiles
        key_len = math.ceil(self.total_tiles / num_tiles)

        keys = np.arange(key_len)
        per_frame = np.array_split(contents, key_len)

        self.img_tile_map: dict[str, list[str]] = defaultdict(list)
        for i, tiles_per_frame in enumerate(per_frame):
            k = keys[i]
            for content in tiles_per_frame:
                self.img_tile_map[k].append(content)

        # random.shuffle(keys)
        self.images = keys

        self.total_images.set(len(self.images))
        self.total_objects.set(self.total_tiles)

    def stream_objects(self) -> None:
        super().stream_objects()
        assert self._context is not None

        for key in self.images:
            time_start = time.time()
            if self._stop_event.is_set():
                break

            self.retrieved_images.inc()

            tiles = self.img_tile_map[key]
            elapsed = time.time() - self._start_time
            logger.info(f"Retrieved Image:{key} Tiles:{len(tiles)} @ {elapsed}")

            for tile in tiles:
                parts = tile.split()
                image_path = Path(parts[0])
                try:
                    class_label = ClassLabel(int(parts[1]))
                    class_name = self._class_id_to_name(class_label)
                except ValueError:
                    class_name = ClassName(sys.intern(parts[1]))
                except IndexError:
                    class_name = NEGATIVE_CLASS

                object_id = ObjectId(f"/{class_name}/collection/id/{image_path}")

                image_path = self._data_root / image_path

                try:
                    if image_path.suffix == ".npy":
                        content = np.load(image_path)
                    else:
                        tmpfile = io.BytesIO()
                        image = Image.open(image_path).convert("RGB")
                        image.save(tmpfile, format="JPEG", quality=85)
                        content = tmpfile.getvalue()
                except (OSError, ValueError):
                    logger.error(f"Failed to read {object_id}")
                    self.failed_objects.inc()
                    continue

                attributes = self.set_tile_attributes(object_id, class_name)

                self.put_objects(
                    ObjectProvider(
                        object_id,
                        content,
                        HawkAttributeProvider(attributes, image_path, self._resize),
                        class_name,
                    )
                )

            retrieved_tiles = collect_metrics_total(self.retrieved_objects)
            logger.info(f"{retrieved_tiles} / {self.total_tiles} RETRIEVED")
            time_passed = time.time() - time_start
            if time_passed < self._timeout:
                time.sleep(self._timeout - time_passed)
