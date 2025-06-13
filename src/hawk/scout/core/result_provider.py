# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from ...classes import class_name_to_str
from ...proto.messages_pb2 import SendTile, _BoundingBox

if TYPE_CHECKING:
    from ...classes import ClassName
    from ...objectid import ObjectId
    from ..retrieval.retriever import Retriever


class BoundingBox(TypedDict, total=False):
    x: float
    y: float
    w: float
    h: float
    class_name: ClassName
    confidence: float


def bbox_to_protobuf(bbox: BoundingBox) -> _BoundingBox:
    return _BoundingBox(
        x=bbox.get("x", 0.5),
        y=bbox.get("y", 0.5),
        w=bbox.get("w", 1.0),
        h=bbox.get("h", 1.0),
        class_name=class_name_to_str(bbox["class_name"]),
        confidence=bbox["confidence"],
    )


class ResultProvider:
    def __init__(
        self,
        object_id: ObjectId,
        score: float,
        bboxes: list[BoundingBox],
        model_version: int | None = None,
        feature_vector: bytes | None = None,
    ):
        self.id = object_id
        self.score = score
        self.bboxes = bboxes
        self.model_version = model_version
        self.feature_vector = feature_vector

        self.gt = object_id._groundtruth()

    def to_protobuf(
        self, retriever: Retriever, scout_index: int, *, novel_sample: bool = False
    ) -> SendTile:
        oracle_data = retriever.get_oracle_data(self.id)
        groundtruth = retriever.get_groundtruth(self.id)

        return SendTile(
            _objectId=self.id.serialize_oid(),
            scoutIndex=scout_index,
            version=self.model_version,
            feature_vector=self.feature_vector,
            oracle_data=[obj.to_protobuf() for obj in oracle_data],
            inferenced=[bbox_to_protobuf(bbox) for bbox in self.bboxes],
            groundtruth=[bbox_to_protobuf(bbox) for bbox in groundtruth],
            novel_sample=novel_sample,
        )
