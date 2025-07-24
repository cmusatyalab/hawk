# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import io
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from logzero import logger

from ....classes import POSITIVE_CLASS
from ....detection import Detection
from ....hawkobject import HawkObject
from ....objectid import ObjectId
from ..retriever import Retriever, RetrieverConfig
from .kinetics_ds import KineticsDs
from .video_utils import create_gif_from_video_tensor_bytes


class K600RetrieverConfig(RetrieverConfig):
    dataset_root: Path
    frames_per_clip: int
    frame_rate: int
    positive_class_idx: int = 0
    timeout: float = 2.0

    # scout 'N of M' (N index starts at 1)
    N: int = 1
    M: int = 1


# Local class to wrap direct accesses to the index stored in ObjectId
@dataclass(frozen=True)
class K600_ObjectId(ObjectId):
    @classmethod
    def from_idx(cls, index: int) -> K600_ObjectId:
        return cls(str(index))

    @classmethod
    def from_oid(cls, object_id: ObjectId) -> K600_ObjectId:
        try:
            int(object_id.oid)
        except ValueError:
            assert "ObjectId does not contain a valid index"
        return cls(object_id.oid)

    @property
    def index(self) -> int:
        return int(self.oid)


class K600Retriever(Retriever):
    config_class = K600RetrieverConfig
    config: K600RetrieverConfig

    def __init__(self, config: K600RetrieverConfig) -> None:
        super().__init__(config)

        self.ds = KineticsDs(
            root=self.config.dataset_root,
            frames_per_clip=self.config.frames_per_clip,
            step_between_clips=self.config.frames_per_clip,
            frame_rate=self.config.frame_rate,
        )

    def _get_next_objectid(self) -> Iterator[K600_ObjectId]:
        num_videos = len(self.ds)
        video_indexes = list(range(num_videos))

        # Shuffle the index, but keep the random order predictable
        random.seed(20220718)
        random.shuffle(video_indexes)

        # slice based on the number of scouts
        if self.config.M > 1:
            assert self.config.N >= 1
            video_indexes = video_indexes[self.config.N - 1 : -1 : self.config.M]

        yield from map(K600_ObjectId.from_idx, video_indexes)

    def get_next_objectid(self) -> Iterator[K600_ObjectId | None]:
        for object_id in self._get_next_objectid():
            elapsed = time.time() - self._start_time
            logger.info(f"Retrieved video: {object_id} @ {elapsed}")
            yield object_id
            yield None

    def get_ml_data(self, object_id: ObjectId) -> HawkObject:
        index = K600_ObjectId.from_oid(object_id).index
        video = self.ds.get_video(index)
        with io.BytesIO() as f:
            torch.save(video, f)
            content = f.getvalue()
        return HawkObject(content=content, media_type="x-tensor/pytorch")

    def get_oracle_data(self, object_id: ObjectId) -> list[HawkObject]:
        index = K600_ObjectId.from_oid(object_id).index
        video = self.ds.get_video(index)
        gif = create_gif_from_video_tensor_bytes(video)
        return [HawkObject(content=gif, media_type="image/gif")]

    def _is_groundtruth_positive(self, object_id: ObjectId) -> bool:
        index = K600_ObjectId.from_oid(object_id).index
        label = self.ds.get_label(index)
        return label == self.config.positive_class_idx

    def get_groundtruth(self, object_id: ObjectId) -> list[Detection]:
        if not self._is_groundtruth_positive(object_id):
            return []
        return [Detection(class_name=POSITIVE_CLASS)]

    def __len__(self) -> int:
        return len(self.ds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()

    k600_retriever = K600Retriever.from_config(
        {
            "dataset_root": args.dataset_root,
            "frames_per_clip": 30,
            "frame_rate": 5,
            "positive_class_idx": 0,
        },
    )

    id_stream = k600_retriever.get_next_objectid()
    video_id = next(id_stream)
    assert video_id is not None

    ml_data = k600_retriever.get_ml_data(video_id)
    assert ml_data.media_type == "x-tensor/pytorch"

    oracle_data = k600_retriever.get_oracle_data(video_id)[0]
    assert oracle_data.media_type == "image/gif"
    oracle_data.to_file("in_memory_output")

    label = int(k600_retriever._is_groundtruth_positive(video_id))
    print(f"label: {label}, id: {video_id}")
