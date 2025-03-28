# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import io
import pickle
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import numpy.typing as npt
from PIL import Image


# Must be serializable
class AttributeProvider(metaclass=ABCMeta):
    @abstractmethod
    def get_image(self) -> Image.Image:
        pass

    @abstractmethod
    def get_thumbnail_size(self) -> int:
        pass

    @abstractmethod
    def get(self) -> Mapping[str, bytes]:
        pass

    @abstractmethod
    def add(self, attribute: Mapping[str, bytes]) -> None:
        pass


class HawkAttributeProvider(AttributeProvider):
    def __init__(
        self,
        attributes: Dict[str, bytes],
        image_provider: Path,
        resize: bool = True,
    ):
        self._attributes = attributes
        self.thumbnail_size = (256, 256)
        self.resize = resize
        self._image_provider = image_provider
        self._is_npy = image_provider.suffix == ".npy"

        self.thumbnail = io.BytesIO()
        if self._is_npy:
            arr: npt.NDArray[Any] = np.load(self._image_provider)
            pickle.dump(arr, self.thumbnail)
        else:
            image = Image.open(self._image_provider).convert("RGB")
            image = image.copy()

            if self.resize and image.size != self.thumbnail_size:
                # crop to centered square and resize to thumbnail_size
                short_edge = min(image.size)
                left = (image.size[0] - short_edge) // 2
                top = (image.size[1] - short_edge) // 2
                right = left + short_edge
                bottom = top + short_edge

                image = image.crop((left, top, right, bottom))
                image = image.resize(self.thumbnail_size)

            image.save(self.thumbnail, "JPEG")

    def get_image(self) -> Image.Image:
        return Image.open(self._image_provider).convert("RGB")

    def get_thumbnail_size(self) -> int:
        return self.thumbnail.getbuffer().nbytes

    def add(self, attribute: Mapping[str, bytes]) -> None:
        for k, v in attribute.items():
            self._attributes[k] = v

    def get(self) -> Mapping[str, bytes]:
        attributes = dict(self._attributes)
        if self._is_npy:
            # maybe we should store this as thumbnail.npy?
            attributes["thumbnail.jpeg"] = self.thumbnail.getvalue()
        else:
            attributes["thumbnail.jpeg"] = self.thumbnail.getvalue()

        return attributes
