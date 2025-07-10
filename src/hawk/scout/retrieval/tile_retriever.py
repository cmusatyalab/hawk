# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from logzero import logger

from ...classes import ClassLabel
from ...objectid import ObjectId
from .retriever import Retriever, RetrieverConfig
from .retriever_mixins import LegacyRetrieverMixin


class TileRetrieverConfig(RetrieverConfig):
    index_path: Path  # file that contains the index


class TileRetriever(Retriever, LegacyRetrieverMixin):
    config_class = TileRetrieverConfig
    config: TileRetrieverConfig

    def __init__(self, config: TileRetrieverConfig) -> None:
        super().__init__(config)

        index_file = (self.config.data_root / self.config.index_path).resolve()

        # make sure index_file is inside the configured data_root.
        # Path.relative_to raises ValueError when it is not a subtree.
        index_file.relative_to(self.config.data_root)

        self.img_tile_map: dict[str, list[str]] = defaultdict(list)
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

    def get_next_objectid(self) -> Iterator[ObjectId | None]:
        assert self._context is not None

        for key in self.images:
            self._context.log(f"RETRIEVE: File {key}")

            tiles = self.img_tile_map[key]
            delta_t = time.time() - self._start_time
            logger.info(f"Retrieved Image:{key} Tiles:{len(tiles)} @ {delta_t}")

            for tile in tiles:
                file_path, label = tile.split()
                image_path = (
                    Path(file_path).resolve().relative_to(self.config.data_root)
                )
                class_label = ClassLabel(int(label))
                class_name = self._class_id_to_name(class_label)

                yield ObjectId(f"/{class_name}/collection/id/{image_path}")

            yield None
