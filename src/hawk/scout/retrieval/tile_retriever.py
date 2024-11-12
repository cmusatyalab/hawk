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

from ...classes import ClassLabel
from ...proto.messages_pb2 import FileDataset
from ..core.attribute_provider import HawkAttributeProvider
from ..core.object_provider import ObjectProvider
from ..stats import collect_metrics_total
from .retriever import Retriever


class TileRetriever(Retriever):
    def __init__(self, mission_id: str, dataset: FileDataset):
        super().__init__(mission_id)
        self.network = False
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

                class_label = ClassLabel(int(label))
                class_name = self._class_id_to_name(class_label)
                object_id = f"/{class_name}/collection/id/{image_path}"
                attributes = self.set_tile_attributes(object_id, class_name)

                self.put_objects(
                    ObjectProvider(
                        object_id,
                        content,
                        HawkAttributeProvider(
                            attributes, Path(image_path), self._resize
                        ),
                        class_name,
                    )
                )

            retrieved_tiles = collect_metrics_total(self.retrieved_objects)
            logger.info(f"{retrieved_tiles} / {self.total_tiles} RETRIEVED")
            time_passed = time.time() - time_start
            if time_passed < self._timeout:
                time.sleep(self._timeout - time_passed)
