# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from io import BytesIO

from logzero import logger
from PIL import Image

from ...detection import Detection
from ...hawkobject import HawkObject
from ...objectid import LegacyObjectId, ObjectId
from ...rusty import unwrap
from .retriever import ImageRetrieverConfig, RetrieverBase

THUMBNAIL_SIZE = 256


class ThumbnailImageMixin(RetrieverBase):
    """Uses Pillow to derive a thumbnail image from the ML ready data."""

    def get_oracle_data(self, object_id: ObjectId) -> list[HawkObject]:
        ml_object = self.get_ml_data(object_id)
        return [self._crop_and_resize(ml_object, THUMBNAIL_SIZE)]

    def _crop_and_resize(self, ml_object: HawkObject, size: int) -> HawkObject:
        if not ml_object.media_type.startswith("image/"):
            msg = "resizing only works for images"
            raise ValueError(msg)

        with BytesIO(ml_object.content) as f:
            image = Image.open(f).convert("RGB")

            # Image operations are delayed as long as possible, so we actually
            # have to keep the file handle open because we may not actually
            # load the image until after crop and resize have been specified
            # and we're trying to save the final image.

            # crop to centered square
            if image.size[0] != image.size[1]:
                short_edge = min(image.size)
                left = (image.size[0] - short_edge) // 2
                top = (image.size[1] - short_edge) // 2
                right = left + short_edge
                bottom = top + short_edge
                image = image.crop((left, top, right, bottom))

            # resize to 'size x size'
            # image.thumbnail((size, size))
            if image.size[0] != size:
                image = image.resize((size, size))

            # either way, convert to compressed jpeg
            with BytesIO() as tmpfile:
                image.save(tmpfile, format="JPEG", quality=85)
                content = tmpfile.getvalue()

        return HawkObject(content=content, media_type="image/jpeg")


class LegacyRetrieverMixin(ThumbnailImageMixin):
    """Get tile and groundtruth based on the legacy ObjectId format."""

    def get_ml_data(self, object_id: ObjectId) -> HawkObject:
        """Return ML ready tile for inferencing or training."""
        try:
            legacy_id = LegacyObjectId.from_objectid(object_id)
            object_path = unwrap(legacy_id.file_path(self.config.data_root))
            ml_object = HawkObject.from_file(object_path)
        except (AssertionError, FileNotFoundError) as e:
            msg = f"Unable to read {object_id}"
            logger.error(msg)
            raise FileNotFoundError(msg) from e

        if isinstance(self.config, ImageRetrieverConfig) and self.config.resize_tile:
            return self._crop_and_resize(ml_object, self.config.tile_size)

        return ml_object

    def get_groundtruth(self, object_id: ObjectId) -> list[Detection]:
        """Return groundtruth for logging, statistics and scriptlabeler."""
        # only handles classification groundtruth as it assumes the class is
        # stashed in the object id.
        legacy_id = LegacyObjectId.from_objectid(object_id)
        class_name = legacy_id.groundtruth
        if class_name is None:
            return []
        return [Detection(class_name=class_name)]
