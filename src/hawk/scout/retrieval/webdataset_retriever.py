# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""WebDataset retriever is actually a misnomer because we need random (indexed)
access to dataset items, so in reality this is an Indexed WebDataset retriever
(wids), where the index is a list of (webdataset shard, number of items)
tuples, or the json description equivalent.

  {
    "wids_version": 1,
    "name": "Dataset name",
    "description": "Dataset description",
    "shardlist": [
      {
        "url": "http://example.com/dataset/shard-000000.tar",
        "nsamples": 10000
      },
      {
        "url": "http://example.com/dataset/shard-000001.tar",
        "nsamples": 700
      }
    ]
  }
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import wids
from logzero import logger
from pydantic import BaseModel, ValidationError

from ...detection import Detection
from ...hawkobject import MEDIA_TYPES, HawkObject
from ...objectid import ObjectId
from ..context.data_manager_context import DataManagerContext
from .retriever import ImageRetrieverConfig, Retriever
from .retriever_mixins import ThumbnailImageMixin

IMAGE_TYPES = {
    suffix: media_type
    for media_type, suffixes in MEDIA_TYPES.items()
    for suffix in suffixes
    if media_type.startswith("image/")
}


class WebDatasetRetrieverConfig(ImageRetrieverConfig):
    webdataset: str | list[tuple[str, int]]

    # split dataset between workers, set to false if each worker has a
    # pre-sharded local copy of the dataset
    split: bool = True
    cache_dir: Path = Path("wids_cache")

    # parameters for shuffling samples
    chunksize: int = 10000
    seed: int = 0
    shuffle: bool = True
    shufflefirst: bool = False

    # used to slow down the retrieval rate
    tiles_per_frame: int = 200


# structure of the meta (json) object containing groundtruth for each item
class WebDatasetMetaObject(BaseModel):
    groundtruth: list[Detection] = []


class WebDatasetRetriever(Retriever, ThumbnailImageMixin):
    config_class = WebDatasetRetrieverConfig
    config: WebDatasetRetrieverConfig

    def __init__(self, config: WebDatasetRetrieverConfig) -> None:
        super().__init__(config)

        self.dataset = wids.ShardListDataset(
            self.config.webdataset,
            cache_dir=self.config.cache_dir,
        )
        # disable automatic image decoding
        self.dataset.transformations = []

        self.start_index = 0
        self.end_index = len(self.dataset)

    def add_context(self, context: DataManagerContext) -> None:
        super().add_context(context)

        if self.config.split:
            # statically split the dataset among workers
            assert self._context is not None
            nitems = len(self.dataset)
            workers = len(self._context.scouts)
            items_per_scout = math.ceil(nitems / workers)

            worker_index = self._context.scout_index
            self.start_index = worker_index * items_per_scout
            self.end_index = min(self.start_index + items_per_scout, nitems)

        # update total image/tile counters
        self.total_tiles = self.end_index - self.start_index
        num_frames = math.ceil(self.total_tiles / self.config.tiles_per_frame)

        self.total_images.set(num_frames)
        self.total_objects.set(self.total_tiles)

        self.sampler = wids.ChunkedSampler(
            self.dataset,
            num_samples=(self.start_index, self.end_index),
            chunksize=self.config.chunksize,
            seed=self.config.seed,
            shuffle=self.config.shuffle,
            shufflefirst=self.config.shufflefirst,
        )

    def get_next_objectid(self) -> Iterator[ObjectId | None]:
        for i, index in enumerate(self.sampler, start=1):
            yield ObjectId(oid=str(index))

            if i % self.config.tiles_per_frame == 0:
                yield None

    def get_ml_data(self, object_id: ObjectId) -> HawkObject:
        index = int(object_id.oid)
        sample = self.dataset[index]

        # assuming there are only a few items per sample and there is only one image
        for suffix in sample:
            if suffix in IMAGE_TYPES:
                image_data = sample[suffix].getvalue()
                media_type = IMAGE_TYPES[suffix]
                return HawkObject(content=image_data, media_type=media_type)
        raise ValueError("No recognizable image data found in sample {object_id}")

    def get_groundtruth(self, object_id: ObjectId) -> list[Detection]:
        index = int(object_id.oid)
        sample = self.dataset[index]

        try:
            meta_json = sample.get("json", "{}")
            meta = WebDatasetMetaObject.model_validate_json(meta_json)
            return meta.groundtruth
        except ValidationError:
            logger.warning("Invalid JSON metadata for object {object_id}")
            return []
