# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mission import Mission


class MissionManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mission: Mission | None = None

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
                msg = "Mission does not exist"
                raise Exception(msg)
            return self._mission

    def remove_mission(self) -> None:
        with self._lock:
            self._mission = None
