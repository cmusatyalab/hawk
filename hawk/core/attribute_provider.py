# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

from abc import ABCMeta, abstractmethod
from typing import Mapping


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
