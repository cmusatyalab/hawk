# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from logzero import logger
from PIL import Image

from ...hawkobject import HawkObject
from ...objectid import ObjectId
from ...rusty import unwrap
from ..core.result_provider import BoundingBox
from .retriever import RetrieverBase

matplotlib.use("agg")

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
        try:
            object_path = unwrap(object_id._file_path(self._data_root))
            return HawkObject.from_file(object_path)
        except (AssertionError, FileNotFoundError):
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


class LegacyRadarMixin(LegacyRetrieverMixin):
    """Derive Oracle data for a radar dataset."""

    def __init__(self) -> None:
        super().__init__()

        # This probably should be configurable
        self._data_root = Path("/srv/diamond/RADAR_DETECTION")

    def get_oracle_data(self, object_id: ObjectId) -> list[HawkObject]:
        """Create a Range-Doppler heatmap from radar data."""
        try:
            object_path = unwrap(object_id._file_path(self._data_root))
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
