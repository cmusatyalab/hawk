# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import streamlit as st

from hawk.home.label_utils import MissionResults

if TYPE_CHECKING:
    import pandas as pd

ABOUT_TEXT = """\
Hawk is a live learning system that leverages distributed machine learning,
human domain expertise, and edge computing to detect the presence of rare
objects and gather valuable training samples under austere and degraded
conditions.
"""

parser = argparse.ArgumentParser()
parser.add_argument("logdir", type=Path)
args = parser.parse_args()

HOME_MISSION_DIR = args.logdir
SCOUT_MISSION_DIR = Path("hawk-missions")


@dataclass
class Mission:
    result_dir: Path

    @classmethod
    def list(cls) -> list[Mission]:
        return [cls(mission) for mission in sorted(HOME_MISSION_DIR.iterdir())]

    @property
    def name(self) -> str:
        return self.result_dir.name

    @property
    def image_dir(self) -> Path:
        return self.result_dir / "images"

    @property
    def log_file(self) -> Path:
        print(type(self.result_dir / "logs" / "hawk.yml"))
        return self.result_dir / "logs" / "hawk.yml"

    def save_new_labels(self, new_labels: pd.Series[int]) -> None:
        mission_data = MissionResults(self.result_dir, sync_labels=True)
        mission_data.save_new_labels(new_labels)

    def get_data(self) -> pd.DataFrame:
        mission_data = MissionResults(self.result_dir, sync_labels=True)
        mission_data.resync()
        return mission_data.df.pipe(self._augment_data)

    @staticmethod
    def _augment_data(data: pd.DataFrame) -> pd.DataFrame:
        data["unlabeled"] = data.imageLabel.isna()
        return data

    def get_stats(self) -> dict[str, Any]:
        filepath = self.result_dir.joinpath("logs", "mission-stats.json")
        if not filepath.exists():
            return {}

        data: dict[str, Any] = json.loads(filepath.read_text())
        data["last_update"] = filepath.stat().st_mtime
        return data


def reset_mission_state() -> None:
    """Reset any mission specific session_state variables when we switched to a
    different mission"""
    selected_labels = [key for key in st.session_state if isinstance(key, int)]
    for key in selected_labels:
        # del st.session_state[key]
        st.session_state[key] = "?"


def save_state(state: str) -> None:
    """Copy changed state from temporary to permanent key.
    Use in combination with this when a widget is defined,
        st.session_state.foo = st.session_state.get("_foo", default)
        st.widget(..., key="foo", on_change=save_state, args=("foo",))
    """
    st.session_state[f"_{state}"] = st.session_state[state]


def page_header(title: str) -> None:
    """Create common page header/sidebar elements for a page"""
    st.set_page_config(
        page_title=title,
        menu_items={
            "Report a bug": "https://github.com/cmusatyalab/hawk/issues",
            "About": ABOUT_TEXT,
        },
        layout="wide",
    )

    # create "Select Mission" pulldown
    mission: Mission | None = st.session_state.get("mission")
    missions = [None, *Mission.list()]
    try:
        selected_mission = missions.index(mission)
    except ValueError:
        selected_mission = None

    st.sidebar.selectbox(
        "Select Mission",
        missions,
        index=selected_mission,
        format_func=lambda x: x.name if x else "",
        key="mission",
        on_change=reset_mission_state,
    )
