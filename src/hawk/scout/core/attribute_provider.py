# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import io
from abc import ABCMeta, abstractmethod
from typing import Dict, Mapping, Union

from PIL import Image


# Must be serializable
class AttributeProvider(metaclass=ABCMeta):
    @abstractmethod
    def get_image(self):
        pass

    @abstractmethod
    def get_thumbnail_size(self):
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
        image_provider: Union[str, bytes],
        resize=True,
    ):
        self._attributes = attributes
        self.thumbnail_size = (256, 256)
        self.resize = resize
        self._image_provider = image_provider
        self.thumbnail = None
        self.set_thumbnail()

    def set_thumbnail(self):
        image = Image.open(self._image_provider).convert("RGB")
        image = image.copy()
        self.thumbnail = io.BytesIO()
        if self.resize:
            image.resize(self.thumbnail_size)
        image.save(self.thumbnail, "JPEG")
        return

    def get_image(self):
        return Image.open(self._image_provider).convert("RGB")

    def get_thumbnail_size(self):
        return self.thumbnail.getbuffer().nbytes

    def add(self, attribute):
        for k, v in attribute.items():
            self._attributes[k] = v
        return

    def get(self) -> Mapping[str, bytes]:
        attributes = dict(self._attributes)

        image = Image.open(self._image_provider).convert("RGB")
        width, height = image.size

        attributes["thumbnail.jpeg"] = self.thumbnail.getvalue()

        return attributes
