# SPDX-FileCopyrightText: 2024-2026 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from hawk.gui.elements import (
    Mission,
    empty_state,
    load_mission,
    max_confidence,
    mission_stats,
    page_header,
)

mission = load_mission()
mission_active = mission.is_active

page_header("Statistics", mission)

NEGATIVE_COLOR = "#94a3b8"
CLASS_COLORS = ["#3b82f6", "#f59e0b", "#10b981", "#8b5cf6", "#ec4899", "#06b6d4"]

PER_MODEL = ":material/view_timeline: Per model version"
PERCENT = ":material/percent: Percentages"


def labeled_by_class_chart(df: pd.DataFrame) -> None:
    """Display a by-class breakdown of labeled samples."""
    labeled_by_class = df.value_counts(
        subset=["groundtruth", "model_version"],
        sort=False,
    ).unstack()

    with st.container(
        horizontal=True,
        vertical_alignment="center",
        horizontal_alignment="distribute",
    ):
        st.markdown("**Labeled samples by class**")
        chart_config = st.segmented_control(
            "Chart options",
            [PER_MODEL, PERCENT],
            default=[PER_MODEL],
            selection_mode="multi",
            label_visibility="collapsed",
            key="labeled_chart_options",
        )

    if labeled_by_class.empty:
        if mission_active:
            empty_state(
                "Waiting for labeled samples",
                "The chart fills in as samples are labeled.",
                icon=":material/hourglass_empty:",
            )
        else:
            empty_state("No labeled samples", icon=":material/label_off:")
        return

    if PER_MODEL not in chart_config:
        # summarize by class
        labeled_by_class = pd.DataFrame(labeled_by_class.T.sum(), columns=["all"])
    if PERCENT in chart_config:
        # scale to percentage of total
        labeled_by_class /= labeled_by_class.sum()
    elif "negative" in labeled_by_class.index:
        labeled_by_class.loc["negative"] *= -1

    chart_data = labeled_by_class.T
    palette = iter(CLASS_COLORS)
    colors: list[Any] = [
        NEGATIVE_COLOR if str(cls) == "negative" else next(palette, "#64748b")
        for cls in chart_data.columns
    ]
    # with horizontal=True the x and y labels refer to the unrotated chart
    st.bar_chart(
        chart_data,
        horizontal=True,
        color=colors,
        x_label="model version" if PER_MODEL in chart_config else "",
        y_label="share of labeled samples" if PERCENT in chart_config else "samples",
        height=320,
    )
    if PERCENT not in chart_config:
        st.caption("Negatives are drawn to the left of the axis.")


def confidence_chart(df: pd.DataFrame) -> None:
    st.markdown("**Confidence of received samples over time**")
    if df.empty:
        empty_state("No samples received yet", icon=":material/hourglass_empty:")
        return

    df = df.copy()
    correct = df["class_name"] == df["groundtruth"]
    df["Label matches"] = df[correct]["confidence"]
    df["Label differs"] = df[~correct & df["groundtruth"].notna()]["confidence"]
    df["Not labeled"] = df[df["groundtruth"].isna()]["confidence"]
    st.scatter_chart(
        df,
        x="time_queued",
        y=["Label matches", "Label differs", "Not labeled"],
        color=["#10b981", "#ef4444", "#94a3b8"],
        x_label="time received",
        y_label="confidence",
        height=320,
    )


@st.fragment(run_every="2s" if mission_active else None)
def display_stats(mission: Mission) -> None:
    mission.resync()
    df = max_confidence(mission.df)

    mission_stats(mission, df)
    st.space("small")

    col1, col2 = st.columns(2, gap="medium")
    with col1, st.container(border=True, height="stretch"):
        labeled_by_class_chart(df)
    with col2, st.container(border=True, height="stretch"):
        confidence_chart(df)

    if mission_active and not mission.is_active:
        st.rerun()


LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


@st.fragment(run_every="10s" if mission_active else None)
def display_logs(mission: Mission) -> None:
    mission_log = mission.mission_dir / "hawk_home.log"
    if not mission_log.exists():
        return

    with st.container(border=True):
        with st.container(
            horizontal=True,
            vertical_alignment="center",
            horizontal_alignment="distribute",
        ):
            st.markdown("**Mission log**")
            with st.container(horizontal=True, width="content", gap="small"):
                levels = st.pills(
                    "Levels",
                    LOG_LEVELS[1:],
                    default=LOG_LEVELS[1:],
                    selection_mode="multi",
                    label_visibility="collapsed",
                    key="log_levels",
                )
                search = st.text_input(
                    "Search log",
                    placeholder="Filter messages…",
                    icon=":material/search:",
                    label_visibility="collapsed",
                    width=220,
                    key="log_search",
                )

        log = pd.read_json(mission_log, lines=True)
        if log.empty:
            st.caption("The log is empty.")
            return

        log = log.set_index("asctime").sort_index(ascending=False)
        log = log[log["levelname"].isin(levels or LOG_LEVELS)]
        if search:
            log = log[log["message"].str.contains(search, case=False, regex=False)]

        st.dataframe(
            log,
            column_order=("levelname", "message"),
            column_config={
                "_index": st.column_config.TextColumn("Time", width="medium"),
                "levelname": st.column_config.TextColumn("Level", width="small"),
                "message": st.column_config.TextColumn("Message", width="large"),
            },
            height=320,
        )


display_stats(mission)
display_logs(mission)
