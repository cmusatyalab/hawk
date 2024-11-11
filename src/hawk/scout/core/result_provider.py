# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
    from ..retrieval.retriever import Retriever
    from .object_provider import ObjectProvider


class BoundingBox(TypedDict, total=False):
    x: float
    y: float
    w: float
    h: float
    class_name: str
    confidence: float


class ResultProvider:
    def __init__(
        self,
        obj: ObjectProvider,
        score: float,
        bboxes: list[BoundingBox],
        model_version: int | None = None,
        feature_vector: torch.Tensor | None = None,
    ):
        self.id = obj.id
        self.attributes = obj.attributes
        self.gt = obj.gt
        self.score = score
        self.bboxes = bboxes
        self.model_version = model_version
        self.feature_vector = feature_vector

    def read_object(self, retriever: Retriever) -> bytes | None:
        obj = retriever.read_object(self.id)
        return obj.content if obj is not None else None
