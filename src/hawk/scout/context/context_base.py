# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Abstract class for mission context
"""

from __future__ import annotations

import time
from abc import ABCMeta, abstractmethod

from ..core.class_manager import MLClassManager
from ..core.hawk_stub import HawkStub


class ContextBase(metaclass=ABCMeta):
    # Time when the mission was started
    start_time: float

    @property
    @abstractmethod
    def scout_index(self) -> int:
        """Index of the scout"""
        pass

    @property
    @abstractmethod
    def scouts(self) -> list[HawkStub]:
        """List of connections to other participating scouts"""
        pass

    def mission_time(self, end_t: float | None = None) -> float:
        """Compute time elapsed since mission was started"""
        if end_t is None:
            end_t = time.time()
        return end_t - self.start_time
