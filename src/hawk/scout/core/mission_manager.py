# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import threading
from typing import Optional

from .mission import Mission


class MissionManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mission: Optional[Mission] = None

    def set_mission(self, mission: Mission) -> None:
        with self._lock:
            if self._mission is not None:
                old_mission = self._mission
                old_mission.stop()
                self.remove_mission()
            self._mission = mission

    def get_mission(self) -> Mission:
        with self._lock:
            if self._mission is None:
                raise Exception("Mission does not exist")
            return self._mission

    def remove_mission(self) -> None:
        with self._lock:
            self._mission = None
        return None
