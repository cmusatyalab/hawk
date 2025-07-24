# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from io import BytesIO

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from logzero import logger
from PIL import Image

from ...hawkobject import HawkObject
from ...objectid import LegacyObjectId, ObjectId
from ...rusty import unwrap
from .random_retriever import RandomRetriever, RandomRetrieverConfig

mpl.use("agg")


class RadarRetriever(RandomRetriever):
    config_class = RandomRetrieverConfig
    config: RandomRetrieverConfig

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
            data.sum(axis=2).transpose(),
            cmap="viridis",
            interpolation="nearest",
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
        stereo_path = object_path.parent.parent.joinpath(
            "canvas",
            object_path.name,
        ).with_suffix(".png")

        if stereo_path.exists():
            stereo_image = Image.open(stereo_path).convert("RGB")

            # crop the stereo RGB image from the combined stereo + RD map canvas
            assert stereo_image.size == (2100, 1500)
            stereo_image = stereo_image.crop((263, 480, 1003, 1035))

            with BytesIO() as tmpfile:
                stereo_image.save(tmpfile, format="JPEG", quality=85)
                content = tmpfile.getvalue()

            oracle_data.append(HawkObject(content=content, media_type="image/jpeg"))

        return oracle_data
