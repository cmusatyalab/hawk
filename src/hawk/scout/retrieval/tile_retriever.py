# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import io
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from logzero import logger
from PIL import Image

from ...proto.messages_pb2 import FileDataset
from ..core.attribute_provider import HawkAttributeProvider
from ..core.object_provider import ObjectProvider
from .retriever import Retriever


class TileRetriever(Retriever):
    def __init__(self, dataset: FileDataset):
        super().__init__()

        self._dataset = dataset
        self._timeout = dataset.timeout
        self._resize = dataset.resizeTile
        index_file = Path(self._dataset.dataPath)
        self.img_tile_map: Dict[str, List[str]] = defaultdict(list)

        self.images = []
        contents = index_file.read_text().splitlines()
        for line in contents:
            # line = "<path> <label>"
            path, _ = line.split()
            key = Path(path).name.split("_")[0]
            if key not in self.img_tile_map:
                self.images.append(key)
            self.img_tile_map[key].append(line)

        self.total_tiles = len(contents)
        self._stats.total_objects = self.total_tiles
        self._stats.total_images = len(self.images)

    def stream_objects(self) -> None:
        super().stream_objects()

        for key in self.images:
            time_start = time.time()
            if self._stop_event.is_set():
                break
            self._stats.retrieved_images += 1
            self._context.log(f"RETRIEVE: File {key}")

            tiles = self.img_tile_map[key]
            delta_t = time.time() - self._start_time
            logger.info(f"Retrieved Image:{key} Tiles:{len(tiles)} @ {delta_t}")

            for tile in tiles:
                tmpfile = io.BytesIO()
                image_path, label = tile.split()
                image = Image.open(image_path).convert("RGB")
                image.save(tmpfile, format="JPEG", quality=85)
                content = tmpfile.getvalue()

                object_id = f"/{label}/collection/id/{image_path}"
                attributes = self.set_tile_attributes(object_id, label)
                self._stats.retrieved_tiles += 1

                self.result_queue.put_nowait(
                    ObjectProvider(
                        object_id,
                        content,
                        HawkAttributeProvider(
                            attributes, Path(image_path), self._resize
                        ),
                        int(label),
                    )
                )
            logger.info(f"{self._stats.retrieved_tiles} / {self.total_tiles} RETRIEVED")
            time_passed = time.time() - time_start
            if time_passed < self._timeout:
                time.sleep(self._timeout - time_passed)
