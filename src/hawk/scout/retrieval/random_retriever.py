# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

from ...classes import NEGATIVE_CLASS, ClassLabel, ClassName
from ...objectid import ObjectId
from .retriever import ImageRetrieverConfig, Retriever
from .retriever_mixins import LegacyRetrieverMixin


class RandomRetrieverConfig(ImageRetrieverConfig):
    index_path: Path  # file that contains the index
    tiles_per_frame: int = 200  # tiles per image


class RandomRetriever(Retriever, LegacyRetrieverMixin):
    config_class = RandomRetrieverConfig
    config: RandomRetrieverConfig

    def __init__(self, config: RandomRetrieverConfig) -> None:
        super().__init__(config)

        index_file = (self.config.data_root / self.config.index_path).resolve()
        self._index_root = index_file.parent.parent

        # make sure _index_root is inside the configured data_root, this way we
        # can be somewhat confident both the index file and the file paths we
        # construct will be inside the allowed file sub tree.
        # Path.relative_to raises ValueError when it is not a subtree.
        self._index_root.relative_to(self.config.data_root)

        self.tiles = index_file.read_text().splitlines()
        self.total_tiles = len(self.tiles)
        num_frames = math.ceil(self.total_tiles / self.config.tiles_per_frame)

        keys = list(range(num_frames))
        # random.shuffle(keys)
        self.images = keys

        self.total_images.set(num_frames)
        self.total_objects.set(self.total_tiles)

    def get_next_objectid(self) -> Iterator[ObjectId | None]:
        for frame in self.images:
            frame_index = frame * self.config.tiles_per_frame
            for tile_offset in range(self.config.tiles_per_frame):
                tile_index = frame_index + tile_offset
                try:
                    tile = self.tiles[tile_index]
                    yield self._tile_to_objectid(tile)
                except IndexError:
                    break
            # Indicate we've finished "tiling" an image
            yield None

    def _tile_to_objectid(self, tile: str) -> ObjectId:
        file_path, groundtruth = (tile.rsplit(maxsplit=1) + ["0"])[:2]
        image_path = (
            self._index_root.joinpath(file_path)
            .resolve()
            .relative_to(self.config.data_root)
        )
        try:
            class_label = ClassLabel(int(groundtruth))
            class_name = self._class_id_to_name(class_label)
        except ValueError:
            class_name = ClassName(groundtruth)
        except IndexError:
            class_name = NEGATIVE_CLASS

        return ObjectId(f"/{class_name}/collection/id/{image_path}")
