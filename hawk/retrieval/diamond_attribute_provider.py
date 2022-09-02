# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import io
from PIL import Image
from typing import Mapping, Dict, Union

from hawk.core.attribute_provider import AttributeProvider
from hawk.core.utils import IntegerAttributeCodec

INT_CODEC = IntegerAttributeCodec()


class DiamondAttributeProvider(AttributeProvider):

    def __init__(self, attributes: Dict[str, bytes], image_provider: Union[str, bytes], resize=True):
        self._attributes = attributes

        for attribute in ['_rows.int', '_cols.int', 'thumbnail.jpeg']:
            if attribute in self._attributes:
                del self._attributes[attribute]

        self.thumbnail_size = (256, 256)
        # self.resize = resize
        self.resize = True
        self._image_provider = image_provider
        self.thumbnail = None
        self.set_thumbnail()

    def set_thumbnail(self):
        image = Image.open(self._image_provider).convert('RGB')
        image = image.copy()
        self.thumbnail = io.BytesIO()
        if self.resize:
            image.resize(self.thumbnail_size)
        image.save(self.thumbnail, 'JPEG')
        return

    def get_image(self):
        return Image.open(self._image_provider).convert('RGB')

    def get_thumbnail_size(self):
        return self.thumbnail.getbuffer().nbytes

    def add(self, attribute):
        for k, v in attribute.items():
            self._attributes[k] = v
        return

    def get(self) -> Mapping[str, bytes]:
        attributes = dict(self._attributes)

        image = Image.open(self._image_provider).convert('RGB')
        width, height = image.size

        attributes['_rows.int'] = INT_CODEC.encode(height)
        attributes['_cols.int'] = INT_CODEC.encode(width)
        attributes['thumbnail.jpeg'] = self.thumbnail.getvalue()

        return attributes
