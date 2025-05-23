# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from PIL import Image

from ...classes import NEGATIVE_CLASS, ClassName
from ...rusty import unwrap

if TYPE_CHECKING:
    from ...objectid import ObjectId
    from ..retrieval.retriever import Retriever
    from .attribute_provider import AttributeProvider
    from .result_provider import ResultProvider


class ObjectProvider:
    def __init__(
        self,
        obj_id: ObjectId,
        content: bytes | npt.NDArray[Any],
        attributes: AttributeProvider,
        gt: ClassName | None = None,
    ):
        self.id = obj_id
        self.content = content
        self.attributes = attributes
        self.gt = gt if gt is not None else NEGATIVE_CLASS

    @classmethod
    def from_result_provider(
        cls, result: ResultProvider, retriever: Retriever
    ) -> ObjectProvider | None:
        obj = result.read_object(retriever)
        if obj is None:
            return None

        is_numpy = unwrap(result.id._file_path()).suffix == ".npy"
        content = cls.load_content(obj, is_numpy)

        return cls(
            result.id,
            content,
            result.attributes,
            result.gt,
        )

    @staticmethod
    def load_content(content: bytes, is_numpy: bool) -> bytes | npt.NDArray[Any]:
        with io.BytesIO(content) as image_file:
            if is_numpy:
                return cast(npt.NDArray[Any], np.load(image_file))

            with io.BytesIO() as tmpfile:
                image = Image.open(image_file).convert("RGB")
                image.save(tmpfile, format="JPEG", quality=85)
                return tmpfile.getvalue()
