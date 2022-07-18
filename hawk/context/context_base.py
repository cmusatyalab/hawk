# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

from abc import ABCMeta, abstractmethod
from typing import List

from hawk.core.hawk_stub import HawkStub


class ContextBase(metaclass=ABCMeta):

    @property
    @abstractmethod
    def scout_index(self) -> int:
        pass

    @property
    @abstractmethod
    def scouts(self) -> List[HawkStub]:
        pass
