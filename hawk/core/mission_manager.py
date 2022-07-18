# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import threading
from typing import Union

from hawk.core.mission import Mission


class MissionManager(object):

    def __init__(self):
        self._lock = threading.Lock()
        self._mission: Union[Mission, None] = None

    def set_mission(self, mission: Mission) -> None:
        with self._lock:
            assert self._mission is None
            self._mission = mission

    def get_mission(self) -> Mission:
        with self._lock:
            return self._mission

    def remove_mission(self) -> None:
        with self._lock:
            self._mission = None

        return None
