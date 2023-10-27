# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import io
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, Mapping

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
        self.thumbnail = io.BytesIO()
        self.set_thumbnail()

    def set_thumbnail(self) -> None:
        image = Image.open(self._image_provider).convert("RGB")
        image = image.copy()
        if self.resize:
            image.resize(self.thumbnail_size)
        self.thumbnail = io.BytesIO()
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

        image = Image.open(self._image_provider).convert("RGB")
        width, height = image.size

        attributes["thumbnail.jpeg"] = self.thumbnail.getvalue()

        return attributes
