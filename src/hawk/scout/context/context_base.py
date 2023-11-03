# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Abstract class for mission context
"""

import time
from abc import ABCMeta, abstractmethod
from typing import List, Optional

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
    def scouts(self) -> List[HawkStub]:
        """List of connections to other participating scouts"""
        pass

    def mission_time(self, end_t: Optional[float] = None) -> float:
        """Compute time elapsed since mission was started"""
        if end_t is None:
            end_t = time.time()
        return end_t - self.start_time
