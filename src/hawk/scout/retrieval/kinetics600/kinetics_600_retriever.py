# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from pandas import DataFrame

from ....classes import POSITIVE_CLASS
from ....hawkobject import HawkObject
from ....objectid import ObjectId
from ...core.result_provider import BoundingBox
from ..retriever import Retriever, RetrieverConfig
from .kinetics_ds import KineticsDs
from .video_utils import create_gif_from_video_tensor_bytes


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


class K600RetrieverConfig(RetrieverConfig):
    root: Path
    frames_per_clip: int
    frame_rate: int
    positive_class_idx: int = 0


class K600Retriever(Retriever):
    config_class = K600RetrieverConfig
    config: K600RetrieverConfig

    def __init__(self, config: K600RetrieverConfig) -> None:
        super().__init__(config)

        self.ds = KineticsDs(
            root=self.config.root,
            video_clips_pkl_name="train.pkl",
            frames_per_clip=self.config.frames_per_clip,
            step_between_clips=self.config.frames_per_clip,
            split="train",
            frame_rate=self.config.frame_rate,
        )

    def get_next_objectid(self) -> Iterator[K600_ObjectId]:
        ## Generator - note that the videos order would be random
        ## and different at for each generator returned from this method.
        num_videos = len(self.ds)
        video_indexes = [K600_ObjectId.from_idx(n) for n in range(num_videos)]
        random.shuffle(video_indexes)
        yield from video_indexes

    def get_ml_data(self, object_id: ObjectId) -> HawkObject:
        index = K600_ObjectId.from_oid(object_id).index
        video = self.ds[index]
        with io.BytesIO() as f:
            torch.save(video, f)
            content = f.getvalue()
        return HawkObject(content=content, media_type="x-tensor/pytorch")

    def get_oracle_data(self, object_id: ObjectId) -> list[HawkObject]:
        index = K600_ObjectId.from_oid(object_id).index
        video = self.ds[index]
        gif = create_gif_from_video_tensor_bytes(video)
        return [HawkObject(content=gif, media_type="image/gif")]

    def is_groundtruth_positive(self, object_id: ObjectId) -> bool:
        index = K600_ObjectId.from_oid(object_id).index
        label = self.ds.get_label(index)
        return label == self.config.positive_class_idx

    def get_groundtruth(self, object_id: ObjectId) -> list[BoundingBox]:
        if not self.is_groundtruth_positive(object_id):
            return []
        return [
            BoundingBox(
                x=0.5, y=0.5, w=1.0, h=1.0, class_name=POSITIVE_CLASS, confidence=1.0
            )
        ]

    def __len__(self) -> int:
        return len(self.ds)

    def generate_index_files(self, num_scouts: int, path: str) -> list[DataFrame]:
        assert num_scouts > 0
        shard_size = len(self) // num_scouts
        id_stream = self.get_next_objectid()
        res: list[DataFrame] = []
        for _shard_id in range(num_scouts - 1):
            scout_index: dict[int, int] = dict()  # video_id.index -> label
            for _ in range(shard_size):
                video_id = next(id_stream)
                scout_index[video_id.index] = int(
                    self.is_groundtruth_positive(video_id)
                )
                assert scout_index[video_id.index] in {0, 1}
            res.append(
                DataFrame.from_dict(scout_index, orient="index", columns=["label"])
            )
        scout_index = dict()
        for video_id in id_stream:
            scout_index[video_id.index] = int(self.is_groundtruth_positive(video_id))
            assert scout_index[video_id.index] in {0, 1}
        res.append(DataFrame.from_dict(scout_index, orient="index", columns=["label"]))
        output_path = Path(path)
        for shard_id, shard in enumerate(res):
            shard.to_csv(f"{path}/scout_{shard_id}.csv")
            shard.to_csv(output_path / f"scout_{shard_id}.csv", index_label="video_id")
        return res


if __name__ == "__main__":
    k600_retriever = K600Retriever.from_config(
        dict(
            root="/home/gil/data/k600",
            frames_per_clip=30,
            frame_rate=5,
            positive_class_idx=0,
        )
    )
    k600_retriever.generate_index_files(num_scouts=1, path="./")
    id_stream = k600_retriever.get_next_objectid()
    video_id = next(id_stream)
    ml_data = k600_retriever.get_ml_data(video_id)
    oracle_data = k600_retriever.get_oracle_data(video_id)[0]
    assert ml_data.media_type == "x-tensor/pytorch"
    assert oracle_data.media_type == "image/gif"
    oracle_data.to_file("in_memory_output")

    label = int(k600_retriever.is_groundtruth_positive(video_id))
    print(f"label: {label}, id: {video_id}")
