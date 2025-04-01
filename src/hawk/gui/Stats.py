# SPDX-FileCopyrightText: 2024-2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from hawk.gui.elements import Mission, load_mission, max_confidence, mission_stats

st.title("Hawk Mission Statistics")

mission = load_mission()
mission_active = mission.state() in ["Starting", "Running"]

st.fragment(run_every="2s" if mission_active else None)


def display_stats(mission: Mission) -> None:
    start = time.time()

    mission.resync()
    df = max_confidence(mission.df)

    # with st.expander("Dataframe"):
    #     st.dataframe(df)

    with st.expander("Overall mission statistics", expanded=True):
        mission_stats(mission, df)

    ####
    # Display a by-class breakdown of labeled samples
    labeled_by_class = df.value_counts(
        subset=["groundtruth", "model_version"],
        sort=False,
    ).unstack()

    if not labeled_by_class.empty:
        with st.expander("Breakdown of labeled samples by class", expanded=True):
            chart_options = [":material/view_timeline:", ":material/percent:"]
            chart_config = st.segmented_control(
                "Chart Options",
                chart_options,
                default=chart_options,
                selection_mode="multi",
                label_visibility="hidden",
            )
            if chart_options[0] not in chart_config:
                # summarize by class
                labeled_by_class = pd.DataFrame(labeled_by_class.T.sum())
            if chart_options[1] in chart_config:
                # scale to percentage of total
                labeled_by_class /= labeled_by_class.sum()
            elif "negative" in labeled_by_class.index:
                labeled_by_class.loc["negative"] *= -1
            st.bar_chart(labeled_by_class.T, horizontal=True, y_label="model version")
    elif mission_active:
        st.info("Waiting for labeled samples")
    else:
        st.error("No labeled samples")

    with st.expander("Confidence for received samples over time"):
        df["correct"] = df["class_name"] == df["groundtruth"]
        df["label_differs"] = df[~df["correct"]]["confidence"]
        df["label_matches"] = df[df["correct"]]["confidence"]
        st.scatter_chart(df, x="time_queued", y=["label_differs", "label_matches"])

    if mission_active and mission.state() == "Finished":
        st.rerun()

    elapsed = time.time() - start
    st.caption(f"Time to render fragment {elapsed:.3f}s\n\n---")


@st.fragment(run_every="10s" if mission_active else None)
def display_logs(mission: Mission) -> None:
    mission_log = mission.mission_dir / "hawk_home.log"
    if mission_log.exists():
        with st.expander("Mission Logs"):
            log = pd.read_json(mission_log, lines=True)
            log.set_index("asctime", inplace=True)

            st.dataframe(
                log,
                column_order=("levelname", "message"),
                column_config={
                    "_index": "timestamp",
                    "levelname": "level",
                    "message": st.column_config.TextColumn(width=None),
                },
            )


display_stats(mission)
display_logs(mission)
