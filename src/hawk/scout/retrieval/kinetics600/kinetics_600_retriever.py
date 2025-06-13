# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pandas import DataFrame
from torch import Tensor

from ....classes import POSITIVE_CLASS
from ....objectid import ObjectId
from ...core.result_provider import BoundingBox
from ..retriever_ifc import RetrieverIfc
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
            int(object_id._id)
        except ValueError:
            assert "ObjectId does not contain a valid index"
        return cls(object_id._id)

    @property
    def index(self) -> int:
        return int(self._id)


class K600Retriever(RetrieverIfc):
    def __init__(
        self,
        root: str,
        frames_per_clip: int,
        frame_rate: int,
        positive_class_idx: int = 0,
    ):
        self.frame_rate: int = frame_rate
        self.frames_per_clip: int = frames_per_clip
        self.root: str = root
        self.positive_class_idx: int = positive_class_idx

        def label_transform(
            y: int, positive_class_idx: int = positive_class_idx
        ) -> int:
            return 1 if y == positive_class_idx else 0

        self.ds = KineticsDs(
            root=root,
            video_clips_pkl_name="train.pkl",
            frames_per_clip=frames_per_clip,
            step_between_clips=frames_per_clip,
            split="train",
            frame_rate=frame_rate,
            label_transform=label_transform,
        )

    def object_ids_stream(self) -> Iterator[K600_ObjectId]:
        ## Generator - note that the videos order would be random
        ## and different at for each generator returned from this method.
        num_videos = len(self.ds)
        video_indexes = [K600_ObjectId.from_idx(n) for n in range(num_videos)]
        random.shuffle(video_indexes)
        yield from video_indexes

    def get_ml_ready_data(self, object_id: ObjectId) -> tuple[Tensor, str]:
        index = K600_ObjectId.from_oid(object_id).index
        video, _ = self.ds[index]
        return video, "x-tensor/pytorch"

    def get_oracle_ready_data(self, object_id: ObjectId) -> list[tuple[bytes, str]]:
        index = K600_ObjectId.from_oid(object_id).index
        video, _ = self.ds[index]
        gif = create_gif_from_video_tensor_bytes(video)
        return [(gif, "image/gif")]

    def get_ground_truth_label(self, object_id: ObjectId) -> int:
        index = K600_ObjectId.from_oid(object_id).index
        return self.ds.get_label(index)

    def get_ground_truth(self, object_id: ObjectId) -> list[BoundingBox]:
        label = self.get_ground_truth_label(object_id)
        if not label:
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
        id_stream = self.object_ids_stream()
        res: list[DataFrame] = []
        for _shard_id in range(num_scouts - 1):
            scout_index: dict[int, int] = dict()  # video_id.index -> label
            for _ in range(shard_size):
                video_id = next(id_stream)
                scout_index[video_id.index] = self.get_ground_truth_label(video_id)
                assert scout_index[video_id.index] in {0, 1}
            res.append(
                DataFrame.from_dict(scout_index, orient="index", columns=["label"])
            )
        scout_index = dict()
        for video_id in id_stream:
            scout_index[video_id.index] = self.get_ground_truth_label(video_id)
            assert scout_index[video_id.index] in {0, 1}
        res.append(DataFrame.from_dict(scout_index, orient="index", columns=["label"]))
        output_path = Path(path)
        for shard_id, shard in enumerate(res):
            shard.to_csv(f"{path}/scout_{shard_id}.csv")
            shard.to_csv(output_path / f"scout_{shard_id}.csv", index_label="video_id")
        return res


if __name__ == "__main__":
    k600_retriever = K600Retriever(
        root="/home/gil/data/k600",
        frames_per_clip=30,
        frame_rate=5,
        positive_class_idx=0,
    )
    k600_retriever.generate_index_files(num_scouts=1, path="./")
    id_stream = k600_retriever.object_ids_stream()
    video_id = next(id_stream)
    video, ml_media_type = k600_retriever.get_ml_ready_data(video_id)
    gif, oracle_media_type = k600_retriever.get_oracle_ready_data(video_id)[0]
    assert ml_media_type == "x-tensor/pytorch"
    assert oracle_media_type == "image/gif"
    with open("in_memory_output.gif", "wb") as f:
        f.write(gif)
    label = k600_retriever.get_ground_truth_label(video_id)
    print(f"label: {label}, id: {video_id}")
