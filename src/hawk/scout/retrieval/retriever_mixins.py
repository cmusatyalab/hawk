# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from logzero import logger
from PIL import Image

from ...hawkobject import HawkObject
from ...objectid import LegacyObjectId, ObjectId
from ...rusty import unwrap
from ..core.result_provider import BoundingBox
from .retriever import ImageRetrieverConfig, RetrieverBase

matplotlib.use("agg")

THUMBNAIL_SIZE = 256


class ThumbnailImageMixin(RetrieverBase):
    """Uses Pillow to derive a thumbnail image from the ML ready data."""

    def get_oracle_data(self, object_id: ObjectId) -> list[HawkObject]:
        ml_object = self.get_ml_data(object_id)
        return [self._crop_and_resize(ml_object, THUMBNAIL_SIZE)]

    def _crop_and_resize(self, ml_object: HawkObject, size: int) -> HawkObject:
        if not ml_object.media_type.startswith("image/"):
            raise ValueError("resizing only works for images")

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

    def get_groundtruth(self, object_id: ObjectId) -> list[BoundingBox]:
        """Return groundtruth for logging, statistics and scriptlabeler."""
        # only handles classification groundtruth as it assumes the class is
        # stashed in the object id.
        legacy_id = LegacyObjectId.from_objectid(object_id)
        class_name = legacy_id.groundtruth
        if class_name is None:
            return []

        return [
            BoundingBox(
                x=0.5, y=0.5, w=1.0, h=1.0, class_name=class_name, confidence=1.0
            )
        ]


class LegacyRadarMixin(LegacyRetrieverMixin):
    """Derive Oracle data for a radar dataset."""

    def get_oracle_data(self, object_id: ObjectId) -> list[HawkObject]:
        """Create a Range-Doppler heatmap from radar data."""
        try:
            legacy_id = LegacyObjectId.from_objectid(object_id)
            object_path = unwrap(legacy_id.file_path(self.config.data_root))
            assert object_path.suffix in (".npy", ".npz")
            data = np.load(object_path, allow_pickle=True)
        except (AssertionError, FileNotFoundError):
            logger.error(f"Unable to read {object_id}")
            return []

        # Create RD map
        plt.imshow(
            data.sum(axis=2).transpose(), cmap="viridis", interpolation="nearest"
        )
        plt.xticks([0, 16, 32, 48, 63], ["-13", "-6.5", "0", "6.5", "13"], fontsize=8)
        plt.yticks([0, 64, 128, 192, 255], ["50", "37.5", "25", "12.5", "0"])
        plt.xlabel("velocity (m/s)")
        plt.ylabel("range (m)")
        # plt.title("RD Map")
        with BytesIO() as rdmap:
            plt.savefig(rdmap, format="png", bbox_inches="tight")
            rdmap_content = rdmap.getvalue()
        plt.close("all")

        oracle_data = [HawkObject(content=rdmap_content, media_type="image/png")]

        # Locate stereo image
        object_base = object_path.stem.split("_", 1)[0]
        stereo_image = object_path.parent / "stereo_left" / f"{object_base}_left.jpg"
        if stereo_image.exists():
            oracle_data.append(HawkObject.from_file(stereo_image))

        return oracle_data
