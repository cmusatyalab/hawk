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

import numpy as np
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

MissionState = Literal["Not Started", "Starting", "Training", "Running", "Finished"]

BLINKENLIGHTS_STATES = {
    "configuring": "grey",
    "bootstrapping": "violet",
    "configured": "blue",
    "inferencing": "green",
    "training": "orange",
    "reexamining": "red",
    "finished": "gray",
}
BLINKENLIGHTS_HELP = """\
- :grey-badge[:material/mystery:] Configuring Scout
- :violet-badge[:material/mystery:] Training Bootstrap model
- :blue-badge[:material/mystery:] Waiting to start mission
- :green-badge[:material/mystery:] Inferencing
- :orange-badge[:material/mystery:] Training new model
- :red-badge[:material/mystery:] Re-scoring top results with new model
- :grey-badge[:material/mystery:] Mission finished
"""


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

    def blinkenlights(self) -> None:
        stats = self.get_stats()
        colors = [
            BLINKENLIGHTS_STATES.get(state, "primary")
            for state in stats.get("mission_state", [])
        ]
        st.markdown(
            "".join([f":{color}-badge[:material/mystery:]" for color in colors]),
            help=BLINKENLIGHTS_HELP,
        )

    def state(self) -> MissionState:
        """Try to derive mission state by looking at a log/stats directory."""
        started = self.mission_dir.joinpath("logs").exists()
        running = self.stats_file.exists()
        active = deployment.check_home(self.mission_dir)

        if not started and not active:
            return "Not Started"
        elif not running and active:
            return "Starting"
        elif running and self.get_stats().get("training", 0):
            return "Training"
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
        df["groundtruth"] = labeled["class_name"].astype(df["class_name"].dtype)

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


def max_confidence(df: pd.DataFrame) -> pd.DataFrame:
    # filter down to just the maximum confidence inferences
    max_conf_idx = df.groupby(["instance", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])[
        "confidence"
    ].idxmax()
    return df.iloc[max_conf_idx]


def mission_stats(mission: Mission, df: pd.DataFrame | None) -> None:
    """Output various stats showing mission progress."""
    if df is None:
        df = max_confidence(mission.df)

    start_time = df.time_queued.min()
    last_update = df.time_queued.max()
    time_elapsed = (last_update - start_time).total_seconds()
    model_version = df.model_version.max() if not df.model_version.empty else -1

    if np.isnan(time_elapsed):
        time_elapsed = 0

    # more complicated than just len(unlabeled) because we're counting received
    # samples, not bounding boxes within a sample, or class scores in a bounding box.
    # so we're counting the number of unique instance values.
    samples_received = df.instance.nunique()

    mission_state = mission.state()

    # read scout stats from logs/mission-stats.json
    stats = mission.get_stats()
    model_version = int(stats.get("version", model_version))
    samples_inferenced = int(stats.get("processedObjects", 0))
    samples_total = int(stats.get("totalObjects", samples_inferenced or 1))

    # compute home stats from received and labeled samples
    total_labeled = df[df["labeled"]].instance.nunique()
    total_by_class = df["groundtruth"].value_counts()

    # st.write(total_by_class)
    negative_labeled = total_by_class["negative"]
    positive_labeled = total_by_class.sum() - negative_labeled
    positive_label_ratio = (
        int(100 * positive_labeled / total_by_class.sum()) if total_labeled else 0
    )

    positive_class_counts = ", ".join(
        f"{cls}: {count}"
        for cls, count in pd.DataFrame(total_by_class[1:]).to_dict()["count"].items()
    )

    received_ratio = (
        int(100 * samples_received / samples_inferenced) if samples_inferenced else 0
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Model Version", model_version)
        st.metric("Samples Inferenced", samples_inferenced)
        st.metric("Samples Received", samples_received)
        st.metric("Received Sample Ratio", f"{received_ratio}%")
        st.metric("Mission Status", mission_state)
    with col2:
        st.metric("Positives Labeled", positive_labeled, help=positive_class_counts)
        st.metric("Negatives Labeled", negative_labeled)
        st.metric("Samples Labeled", total_labeled)
        st.metric("Positive Sample Ratio", f"{positive_label_ratio}%")
        st.metric("Elapsed Mission Time", f"{time_elapsed:.1f}s")

    st.progress(float(samples_inferenced / samples_total))
