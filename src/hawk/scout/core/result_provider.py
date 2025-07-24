# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from typing import TYPE_CHECKING

from ...classes import NEGATIVE_CLASS
from ...objectid import LegacyObjectId
from ...proto.messages_pb2 import SendTile

if TYPE_CHECKING:
    from ...detection import Detection
    from ...objectid import ObjectId
    from ..retrieval.retriever import Retriever


class ResultProvider:
    def __init__(
        self,
        object_id: ObjectId,
        score: float,
        bboxes: list[Detection],
        model_version: int | None = None,
        feature_vector: bytes | None = None,
    ) -> None:
        self.id = object_id
        self.score = score
        self.bboxes = bboxes
        self.model_version = model_version
        self.feature_vector = feature_vector

        try:
            legacy_id = LegacyObjectId.from_objectid(object_id)
            self.gt = legacy_id.groundtruth
        except AssertionError:
            self.gt = NEGATIVE_CLASS

    def to_protobuf(
        self,
        retriever: Retriever,
        scout_index: int,
        *,
        novel_sample: bool = False,
        oracle_mode: bool = False,
    ) -> SendTile:
        oracle_data = retriever.get_oracle_data(self.id)
        groundtruth = retriever.get_groundtruth(self.id)

        # replaced inferenced results with groundtruth if we are running in
        # "oracle_mode"
        inferenced = self.bboxes if not oracle_mode else groundtruth

        return SendTile(
            object_id=self.id.to_protobuf(),
            scoutIndex=scout_index,
            version=self.model_version,
            feature_vector=self.feature_vector,
            oracle_data=[obj.to_protobuf() for obj in oracle_data],
            inferenced=[detection.to_protobuf() for detection in inferenced],
            groundtruth=[detection.to_protobuf() for detection in groundtruth],
            novel_sample=novel_sample,
        )
