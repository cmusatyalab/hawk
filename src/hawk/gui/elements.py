# SPDX-FileCopyrightText: 2024-2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal

import pandas as pd
import streamlit as st
from blinker import Signal, signal

from hawk.gui import deployment
from hawk.home.label_utils import DetectionDict, LabelSample, MissionData, read_jsonl
from hawk.mission_config import MissionConfig, load_config

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

HOME_MISSION_DIR = Path(os.environ.get("HAWK_MISSION_DIR", Path.cwd()))
SCOUT_MISSION_DIR = Path("hawk-missions")

MissionState = Literal["Not Started", "Starting", "Running", "Finished"]


@dataclass
class Mission(MissionData):
    @classmethod
    def missions(cls) -> list[str]:
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
        return self.mission_dir.name

    @property
    def is_template(self) -> bool:
        return self.name.startswith("_")

    @property
    def extra_config_files(self) -> list[str]:
        return [
            file
            for file in [
                self.config.get("dataset", {}).get("stream_path"),
                self.config.get("train_strategy", {}).get("bootstrap_path"),
                self.config.get("train_strategy", {}).get("initial_model_path"),
            ]
            if file is not None and self.mission_dir.joinpath(file).exists()
        ]

    @property
    def config(self) -> MissionConfig:
        if not hasattr(self, "_config"):
            try:
                self._config_file = self.mission_dir / "logs" / "hawk.yml"
                self._config = load_config(self._config_file)
                self.config_writable = False
            except FileNotFoundError:
                self._config_file = self.mission_dir / "mission_config.yml"
                self._config = (
                    load_config(self._config_file)
                    if self._config_file.exists()
                    else MissionConfig.from_dict({})
                )
                self.config_writable = True
        return self._config

    @property
    def description(self) -> str:
        return str(self.config.get("description", ""))

    def image_path(self, sample: LabelSample) -> Path:
        return sample.content(self.mission_dir / "images", ".jpeg")

    @property
    def stats_file(self) -> Path:
        return self.mission_dir / "logs" / "mission-stats.json"

    def get_stats(self) -> dict[str, Any]:
        filepath = self.stats_file
        if not filepath.exists():
            return {}

        data: dict[str, Any] = json.loads(filepath.read_text())
        data["last_update"] = filepath.stat().st_mtime
        return data

    def state(self) -> MissionState:
        """Try to derive mission state by looking at a log/stats directory."""
        started = self.mission_dir.joinpath("logs").exists()
        running = self.stats_file.exists()
        active = deployment.check_home(self.mission_dir)

        if not started and not active:
            return "Not Started"
        elif not running and active:
            return "Starting"
        elif running and active:
            return "Running"
        else:  # started/running and not active:
            return "Finished"

    def to_dataframe(self, labels: Iterable[LabelSample]) -> pd.DataFrame:
        image_dir = self.mission_dir / "images"

        # get a list of all class/confidence scores for a bounding box in a sample.
        detections: list[DetectionDict] = []
        for idx, label in enumerate(labels):
            if label.objectId is not None:
                detections.extend(label.to_flat_dict(idx, image_dir))

        # convert the list to a pandas dataframe
        df = pd.DataFrame.from_records(
            detections,
            columns=DetectionDict.__annotations__.keys(),
        ).astype({"class_name": "category"})

        df["time_queued"] = pd.to_datetime(df["time_queued"], unit="s", utc=True)

        # reorder class labels so that the negative class always comes first (index 0)
        if "negative" not in df.class_name.cat.categories:
            df.class_name = df.class_name.cat.add_categories(["negative"])
        positives = [c for c in df.class_name.cat.categories if c != "negative"]
        df.class_name = df.class_name.cat.reorder_categories(["negative", *positives])
        return df

    @property
    def unlabeled_df(self) -> pd.DataFrame:
        return self.to_dataframe(self.unlabeled or read_jsonl(self.unlabeled_jsonl))

    @property
    def labeled_df(self) -> pd.DataFrame:
        return self.to_dataframe(
            self.labeled.values() or read_jsonl(self.labeled_jsonl)
        )

    @property
    def df(self) -> pd.DataFrame:
        df = self.unlabeled_df.set_index(["object_id"])
        labeled = self.labeled_df.set_index(["object_id"])
        df["labeled"] = labeled["confidence"].any()

        df = df.set_index(["bbox_x", "bbox_y", "bbox_w", "bbox_h"], append=True)
        labeled = labeled.set_index(
            ["bbox_x", "bbox_y", "bbox_w", "bbox_h"], append=True
        )
        df["groundtruth"] = labeled["class_name"]

        return df.reset_index()


# cacheable resource?
def load_mission() -> Mission:
    mission_name = st.session_state.get("mission_name")
    if mission_name is not None:
        try:
            mission = Mission.load(mission_name)
            return mission
        except ValueError:
            del st.session_state["mission_name"]

    from hawk.gui.Welcome import welcome_page

    st.switch_page(welcome_page)


def reset_mission_state(sender: Signal | None) -> None:
    """Reset any mission specific session_state variables when we switched to a
    different mission"""
    selected_labels = [key for key in st.session_state if isinstance(key, int)]
    for key in selected_labels:
        # del st.session_state[key]
        st.session_state[key] = "?"


mission_changed = signal("mission-changed")
mission_changed.connect(reset_mission_state)


def save_state(state: str) -> None:
    """Copy changed state from temporary to permanent key.
    Use in combination with this when a widget is defined,
        st.session_state.foo = st.session_state.get("_foo", default)
        st.widget(..., key="foo", on_change=save_state, args=("foo",))
    """
    st.session_state[f"_{state}"] = st.session_state[state]


def columns(ncols: int) -> Iterator[DeltaGenerator]:
    """Generator function to create infinite list of columns"""
    while 1:
        yield from st.columns(ncols)


@contextmanager
def paginate(
    result_list: list[LabelSample], results_per_page: int
) -> Iterator[list[LabelSample]]:
    """Paginate a list of results"""
    nresults = len(result_list)
    current_page = int(st.query_params.get("page", 1))
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
