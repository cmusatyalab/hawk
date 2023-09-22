# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Abstract class for mission context
"""

from abc import ABCMeta, abstractmethod
from typing import List

from ..core.hawk_stub import HawkStub


class ContextBase(metaclass=ABCMeta):
    @property
    @abstractmethod
    def scout_index(self) -> int:
        """Index of the scout"""
        pass

    @property
    @abstractmethod
    def scouts(self) -> List[HawkStub]:
        """List of connections to other participating scouts"""
        pass
