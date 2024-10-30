# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import streamlit as st

from hawk.home.label_utils import LabelSample, MissionResults
from hawk.mission_config import MissionConfig, load_config

ABOUT_TEXT = """\
Hawk is a live learning system that leverages distributed machine learning,
human domain expertise, and edge computing to detect the presence of rare
objects and gather valuable training samples under austere and degraded
conditions.
"""

parser = argparse.ArgumentParser()
parser.add_argument("logdir", type=Path)
args = parser.parse_args()

HOME_MISSION_DIR = args.logdir.resolve()
SCOUT_MISSION_DIR = Path("hawk-missions")


@dataclass
class Mission(MissionResults):
    @classmethod
    def list(cls) -> list[str]:
        return [mission.name for mission in sorted(HOME_MISSION_DIR.iterdir())]

    @classmethod
    def load(cls, mission_name: str) -> Mission:
        mission_path = HOME_MISSION_DIR.joinpath(mission_name).resolve()
        # raises ValueError if we are not a subpath of HOME_MISSION_DIR
        # raises AssertionError if the final name does not match
        assert mission_path.relative_to(HOME_MISSION_DIR).name == mission_name
        return cls(mission_path)

    @property
    def name(self) -> str:
        return Path(self.mission_dir).name

    @property
    def config(self) -> MissionConfig:
        if not hasattr(self, "_config"):
            self._config = load_config(self.config_file)
        return self._config

    def image_path(self, sample: LabelSample) -> Path:
        return sample.unique_name(Path(self.mission_dir) / "images", ".jpeg")

    @property
    def config_file(self) -> Path:
        return Path(self.mission_dir, "logs", "hawk.yml")

    @property
    def stats_file(self) -> Path:
        return Path(self.mission_dir, "logs", "mission-stats.json")

    def get_stats(self) -> dict[str, Any]:
        filepath = self.stats_file
        if not filepath.exists():
            return {}

        data: dict[str, Any] = json.loads(filepath.read_text())
        data["last_update"] = filepath.stat().st_mtime
        return data


# cacheable resource?
def load_mission(mission_name: str | None) -> Mission | None:
    try:
        assert mission_name is not None
        return Mission.load(mission_name)
    except (AssertionError, ValueError):
        return None


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


def page_header(title: str) -> Mission | None:
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
    mission_name: str | None = st.session_state.get("mission", "")
    missions = [None, *Mission.list()]
    try:
        selected_mission = missions.index(mission_name)
    except ValueError:
        selected_mission = None

    mission = st.sidebar.selectbox(
        "Select Mission",
        missions,
        index=selected_mission,
        key="mission",
        on_change=reset_mission_state,
    )
    return load_mission(mission)


@contextmanager
def paginate(result_list: list[LabelSample]) -> Iterator[list[LabelSample]]:
    """Paginate a list of results"""
    nresults = len(result_list)
    current_page = int(st.query_params.get("page", 1))
    results_per_page = st.session_state.rows * st.session_state.columns
    pages = int((nresults + results_per_page - 1) / results_per_page)

    page = max(1, min(pages, current_page))
    if page != current_page:
        st.query_params["page"] = str(page)

    # return slice of the original list based on current page
    start = results_per_page * (page - 1)
    end = start + results_per_page
    yield result_list[start:end]

    if pages <= 1:
        return

    # display pagination
    def goto_page(current_page: int, pages: int) -> None:
        chosen_page = st.session_state.chosen_page
        if chosen_page == "first":
            chosen_page = 1
        if chosen_page == "prev":
            chosen_page = max(1, current_page - 1)
        if chosen_page == "next":
            chosen_page = min(pages, current_page + 1)
        if chosen_page == "last":
            chosen_page = pages
        st.query_params["page"] = str(chosen_page)

    options = (
        ["first", "prev"]
        + list(range(1, current_page)[-5:])
        + [current_page]
        + list(range(current_page + 1, pages + 1)[:5])
        + ["next", "last"]
    )
    st.session_state["chosen_page"] = current_page
    st.radio(
        "Navigate to page",
        options,
        key="chosen_page",
        horizontal=True,
        on_change=goto_page,
        args=(current_page, pages),
    )
