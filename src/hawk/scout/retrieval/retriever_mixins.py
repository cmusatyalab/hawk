# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from logzero import logger
from PIL import Image

from ...hawkobject import HawkObject
from ...objectid import ObjectId
from ..core.result_provider import BoundingBox
from .retriever import RetrieverBase

THUMBNAIL_SIZE = (256, 256)


class ThumbnailImageMixin(RetrieverBase):
    """Uses Pillow to derive a thumbnail image from the ML ready data."""

    def get_oracle_data(self, object_id: ObjectId) -> list[HawkObject]:
        ml_object = self.get_ml_data(object_id)
        if ml_object is None or not ml_object.media_type.startswith("image/"):
            raise ValueError("Generic get_oracle_data only works for images")

        with BytesIO(ml_object.content) as f:
            image = Image.open(f)

        image = image.convert("RGB")

        # crop to centered square
        if image.size[0] != image.size[1]:
            short_edge = min(image.size)
            left = (image.size[0] - short_edge) // 2
            top = (image.size[1] - short_edge) // 2
            right = left + short_edge
            bottom = top + short_edge
            image = image.crop((left, top, right, bottom))

        # resize to THUMBNAIL_SIZE
        # image.thumbnail(THUMBNAIL_SIZE)
        image = image.resize(THUMBNAIL_SIZE)

        with BytesIO() as tmpfile:
            image.save(tmpfile, format="JPEG", quality=85)
            content = tmpfile.getvalue()

        return [HawkObject(content=content, media_type="image/jpeg")]


class LegacyRetrieverMixin(ThumbnailImageMixin):
    """Get tile and groundtruth based on the legacy ObjectId format."""

    def __init__(self) -> None:
        super().__init__()

        # TODO: we probably should be more restrictive about where images
        # are allowed to be retrieved from.
        # Random retriever sets this to the parent of the INDEXES directory.
        self._data_root = Path("/")

    def get_ml_data(self, object_id: ObjectId) -> HawkObject | None:
        """Return ML ready tile for inferencing or training."""
        object_path = object_id._file_path(self._data_root)
        if object_path is None:
            logger.error(f"Unable to get path for {object_id}")
            return None
        try:
            return HawkObject.from_file(object_path)
        except FileNotFoundError:
            logger.error(f"Unable to read {object_id}")
            return None

    def get_groundtruth(self, object_id: ObjectId) -> list[BoundingBox]:
        """Return groundtruth for logging, statistics and scriptlabeler."""
        # only handles classification groundtruth as it assumes it is stashed
        # in the object id.
        class_name = object_id._groundtruth()
        if class_name is None:
            return []

        return [
            BoundingBox(
                x=0.5, y=0.5, w=1.0, h=1.0, class_name=class_name, confidence=1.0
            )
        ]
