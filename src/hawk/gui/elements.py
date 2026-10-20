# SPDX-FileCopyrightText: 2024-2026 Carnegie Mellon University
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
    from streamlit.elements.lib.layout_utils import Gap

HOME_MISSION_DIR = Path(os.environ.get("HAWK_MISSION_DIR", Path.cwd()))
SCOUT_MISSION_DIR = Path("hawk-missions")

MissionState = Literal["Not Started", "Starting", "Training", "Running", "Finished"]
BadgeColor = Literal["red", "orange", "yellow", "blue", "green", "violet", "gray"]

# Per-scout state as reported in logs/mission-stats.json
# state: (badge color, short label, description)
SCOUT_STATES: dict[str, tuple[BadgeColor, str, str]] = {
    "configuring": ("gray", "Configuring", "Configuring scout"),
    "bootstrapping": ("violet", "Bootstrapping", "Training bootstrap model"),
    "configured": ("blue", "Ready", "Waiting for the mission to start"),
    "inferencing": ("green", "Inferencing", "Running inference on new data"),
    "training": ("orange", "Training", "Training a new model"),
    "reexamining": ("red", "Re-examining", "Re-scoring top results with new model"),
    "finished": ("gray", "Finished", "Mission finished"),
}

# Overall mission state: (badge color, icon)
MISSION_STATES: dict[MissionState, tuple[BadgeColor, str]] = {
    "Not Started": ("gray", ":material/radio_button_unchecked:"),
    "Starting": ("blue", ":material/progress_activity:"),
    "Training": ("orange", ":material/model_training:"),
    "Running": ("green", ":material/play_circle:"),
    "Finished": ("violet", ":material/flag:"),
}


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

    def image_path(self, sample: LabelSample, index: int) -> Path:
        return sample.content(self.mission_dir / "images", index=index)

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

    @property
    def is_active(self) -> bool:
        """True while the mission is still producing new results."""
        return self.state() in ("Starting", "Running", "Training")

    def scout_states(self) -> list[tuple[str, str]]:
        """List of (scout host, scout state) tuples."""
        scouts = [scout.host for scout in self.config.deploy.scouts]
        states = self.get_stats().get("mission_state", [])
        states = list(states) + ["configuring"] * (len(scouts) - len(states))
        if len(scouts) < len(states):
            scouts += [f"scout {i}" for i in range(len(scouts), len(states))]
        return list(zip(scouts, states))

    def blinkenlights(self, compact: bool = False) -> None:
        """Show a status badge for every scout in the mission."""
        scout_states = self.scout_states()
        if not scout_states:
            return

        with st.container(horizontal=True, gap="small"):
            for index, (scout, state) in enumerate(scout_states):
                color, label, description = SCOUT_STATES.get(
                    state,
                    ("gray", state.capitalize(), state),
                )
                st.badge(
                    f"S{index}" if compact else f"{scout.split('.')[0]} · {label}",
                    icon=":material/satellite_alt:",
                    color=color,
                    help=f"**{scout}**  \n{description}",
                )

    def state(self) -> MissionState:
        """Try to derive mission state by looking at a log/stats directory."""
        started = self.mission_dir.joinpath("logs").exists()
        running = self.stats_file.exists()
        active = deployment.check_home(self.mission_dir)

        if not started and not active:
            return "Not Started"
        if not running and active:
            return "Starting"
        if running and self.get_stats().get("training", 0):
            return "Training"
        if running and active:
            return "Running"
        # started/running and not active:
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
        ).astype({"class_name": "category", "object_id": "string"})

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
            self.labeled.values() or read_jsonl(self.labeled_jsonl),
        )

    @property
    def df(self) -> pd.DataFrame:
        df = self.unlabeled_df.set_index(["object_id"])
        labeled = self.labeled_df.set_index(["object_id"])
        df["labeled"] = labeled["confidence"].any()

        df = df.set_index(["bbox_x", "bbox_y", "bbox_w", "bbox_h"], append=True)
        labeled = labeled.set_index(
            ["bbox_x", "bbox_y", "bbox_w", "bbox_h"],
            append=True,
        )
        df["groundtruth"] = labeled["class_name"].astype(df["class_name"].dtype)

        return df.reset_index()


# cacheable resource?
def load_mission() -> Mission:
    mission_name = st.session_state.get("mission_name")
    if mission_name is not None:
        try:
            return Mission.load(mission_name)
        except ValueError:
            del st.session_state["mission_name"]

    from hawk.gui.Welcome import welcome_page

    st.switch_page(welcome_page)
    raise AssertionError("shouldn't get here...")


def reset_mission_state(sender: Signal | None) -> None:
    """Reset any mission specific session_state variables when we switched to a
    different mission.
    """
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
        st.widget(..., key="foo", on_change=save_state, args=("foo",)).
    """
    st.session_state[f"_{state}"] = st.session_state[state]


def columns(ncols: int, gap: Gap = "small") -> Iterator[DeltaGenerator]:
    """Generator function to create infinite list of columns."""
    while 1:
        yield from st.columns(ncols, gap=gap)


def state_badge(state: MissionState) -> None:
    """Show the overall mission state as a colored badge."""
    color, icon = MISSION_STATES.get(state, ("gray", ":material/help:"))
    st.badge(state, icon=icon, color=color)


def page_header(
    title: str,
    mission: Mission | None = None,
    icon: str | None = None,
) -> DeltaGenerator:
    """Consistent page header with the mission name and state.

    Returns a right-aligned container that pages can use for their actions.
    """
    with st.container(
        horizontal=True,
        vertical_alignment="center",
        gap="medium",
    ):
        with st.container(gap="xsmall"):
            st.title(title, icon=icon, anchor=False)
            if mission is not None:
                with st.container(horizontal=True, gap="small"):
                    st.badge(
                        mission.name,
                        icon=":material/folder_open:",
                        color="gray",
                    )
                    state_badge(mission.state())
        actions = st.container(
            horizontal=True,
            horizontal_alignment="right",
            vertical_alignment="center",
            width="content",
        )
    return actions


def empty_state(
    title: str,
    body: str = "",
    icon: str = ":material/inbox:",
) -> None:
    """Friendly placeholder for when there is nothing to show yet."""
    with st.container(border=True, horizontal_alignment="center", gap="xsmall"):
        st.space("small")
        st.markdown(f"## :gray[{icon}]", text_alignment="center")
        st.markdown(f"**{title}**", text_alignment="center")
        if body:
            st.caption(body, text_alignment="center")
        st.space("small")


def format_duration(seconds: float) -> str:
    """Human friendly duration, e.g. 1h 04m or 8m 52s."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


@contextmanager
def paginate(
    result_list: list[LabelSample],
    results_per_page: int,
) -> Iterator[list[LabelSample]]:
    """Paginate a list of results.

    The current page is kept in the URL (?page=N) so it survives reloads and
    can be shared.
    """
    nresults = len(result_list)
    pages = max(1, (nresults + results_per_page - 1) // results_per_page)

    if "page" not in st.session_state:
        try:
            st.session_state["page"] = int(st.query_params.get("page", 1))
        except ValueError:
            st.session_state["page"] = 1
    page = max(1, min(pages, st.session_state["page"]))
    st.session_state["page"] = page

    # return slice of the original list based on current page
    start = results_per_page * (page - 1)
    end = start + results_per_page
    yield result_list[start:end]

    if nresults == 0:
        return

    with st.container(
        horizontal=True,
        horizontal_alignment="distribute",
        vertical_alignment="center",
    ):
        st.caption(f"Showing {start + 1}–{min(end, nresults)} of {nresults:,}")
        if pages > 1:
            page = st.pagination(pages, key="page")

    if page > 1:
        st.query_params["page"] = str(page)
    else:
        st.query_params.pop("page", None)


@dataclass
class MissionSummary:
    """Snapshot of mission progress, combining scout and home statistics."""

    state: MissionState
    model_version: int
    samples_total: int
    samples_inferenced: int
    samples_received: int
    positives_labeled: int
    negatives_labeled: int
    positives_by_class: dict[str, int]
    elapsed: float
    has_scout_stats: bool = True

    @property
    def model(self) -> str:
        return f"v{self.model_version}" if self.model_version >= 0 else "—"

    @property
    def labeled(self) -> int:
        return self.positives_labeled + self.negatives_labeled

    @property
    def progress(self) -> float:
        if not self.samples_total:
            return 0.0
        return min(1.0, self.samples_inferenced / self.samples_total)

    @property
    def received_ratio(self) -> float:
        if not self.samples_inferenced:
            return 0.0
        return self.samples_received / self.samples_inferenced

    @property
    def positive_ratio(self) -> float:
        return self.positives_labeled / self.labeled if self.labeled else 0.0


def max_confidence(df: pd.DataFrame) -> pd.DataFrame:
    # filter down to just the maximum confidence inferences
    max_conf_idx = df.groupby(["instance", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])[
        "confidence"
    ].idxmax()
    return df.iloc[max_conf_idx]


def summarize_mission(mission: Mission, df: pd.DataFrame | None) -> MissionSummary:
    if df is None:
        df = max_confidence(mission.df)

    start_time = df.time_queued.min()
    last_update = df.time_queued.max()
    time_elapsed = (last_update - start_time).total_seconds()
    model_version = df.model_version.max() if not df.model_version.empty else -1

    if np.isnan(time_elapsed):
        time_elapsed = 0

    # read scout stats from logs/mission-stats.json
    stats = mission.get_stats()
    samples_inferenced = int(stats.get("processedObjects", 0))

    # compute home stats from received and labeled samples
    total_by_class = df["groundtruth"].value_counts()
    negative_labeled = int(total_by_class.get("negative", 0))
    positives_by_class = {
        str(cls): int(count)
        for cls, count in total_by_class.items()
        if cls != "negative" and count
    }

    return MissionSummary(
        state=mission.state(),
        model_version=int(stats.get("version", model_version)),
        samples_total=int(stats.get("totalObjects", samples_inferenced or 1)),
        samples_inferenced=samples_inferenced,
        # more complicated than just len(unlabeled) because we're counting
        # received samples, not bounding boxes within a sample, or class scores
        # in a bounding box. so we're counting the number of unique instances.
        samples_received=int(df.instance.nunique()),
        positives_labeled=sum(positives_by_class.values()),
        negatives_labeled=negative_labeled,
        positives_by_class=positives_by_class,
        elapsed=time_elapsed,
        has_scout_stats=bool(stats),
    )


def mission_stats(
    mission: Mission,
    df: pd.DataFrame | None,
    compact: bool = False,
) -> MissionSummary:
    """Output various stats showing mission progress."""
    summary = summarize_mission(mission, df)

    positive_class_counts = "  \n".join(
        f"**{cls}**: {count:,}" for cls, count in summary.positives_by_class.items()
    )

    if compact:
        st.progress(
            summary.progress,
            text=f"{summary.samples_inferenced:,} / {summary.samples_total:,}"
            " inferenced",
        )
        items = [
            ("Model", summary.model),
            ("Received", f"{summary.samples_received:,}"),
            ("Positives", f"{summary.positives_labeled:,}"),
            ("Negatives", f"{summary.negatives_labeled:,}"),
        ]
        for column, (label, value) in zip(columns(2), items):
            with column, st.container(gap="xxsmall"):
                st.caption(label)
                st.markdown(f"**{value}**")
        return summary

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Model version",
            summary.model,
            icon=":material/neurology:",
            border=True,
            height="stretch",
        )
    with col2:
        st.metric(
            "Samples inferenced",
            f"{summary.samples_inferenced:,}",
            delta=f"{summary.progress:.0%}" if summary.has_scout_stats else None,
            delta_description=f"of {summary.samples_total:,}"
            if summary.has_scout_stats
            else None,
            delta_color="off",
            delta_arrow="off",
            icon=":material/memory:",
            border=True,
            height="stretch",
        )
    with col3:
        st.metric(
            "Samples received",
            f"{summary.samples_received:,}",
            delta=f"{summary.received_ratio:.2%}",
            delta_description="of inferenced",
            delta_color="off",
            delta_arrow="off",
            icon=":material/download:",
            border=True,
            height="stretch",
        )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Positives labeled",
            f"{summary.positives_labeled:,}",
            delta=f"{summary.positive_ratio:.0%}",
            delta_description="of labeled",
            delta_color="green" if summary.positives_labeled else "off",
            delta_arrow="off",
            help=positive_class_counts or None,
            icon=":material/check_circle:",
            border=True,
            height="stretch",
        )
    with col2:
        st.metric(
            "Negatives labeled",
            f"{summary.negatives_labeled:,}",
            icon=":material/cancel:",
            border=True,
            height="stretch",
        )
    with col3:
        st.metric(
            "Elapsed time",
            format_duration(summary.elapsed),
            icon=":material/schedule:",
            border=True,
            height="stretch",
        )

    if summary.has_scout_stats:
        st.progress(
            summary.progress,
            text=f"Scouts have inferenced {summary.progress:.1%} of the data stream",
        )
    return summary
