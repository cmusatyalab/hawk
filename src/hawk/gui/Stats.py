# SPDX-FileCopyrightText: 2024-2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from hawk.gui.elements import Mission, load_mission


def max_confidence(df: pd.DataFrame) -> pd.DataFrame:
    # filter down to just the maximum confidence inferences
    max_conf_idx = df.groupby(["instance", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])[
        "confidence"
    ].idxmax()
    return df.iloc[max_conf_idx]


st.title("Hawk Mission Statistics")

mission = load_mission()
df = max_confidence(mission.df)

# with st.expander("Dataframe"):
#     st.dataframe(df)


def mission_stats(mission: Mission, df: pd.DataFrame | None) -> bool:
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

    return mission_state in ["Starting", "Running"]


with st.expander("Overall mission statistics", expanded=True):
    mission_active = mission_stats(mission, df)


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
    df["confidence_incorrect"] = df[~df["correct"]]["confidence"]
    df["confidence_correct"] = df[df["correct"]]["confidence"]
    st.scatter_chart(
        df, x="time_queued", y=["confidence_incorrect", "confidence_correct"]
    )

mission_log = mission.mission_dir / "hawk_home.log"
if mission_log.exists():
    with st.expander("Mission Logs"):
        st.text_area("Mission Logs", mission_log.read_text())

if mission_active:
    st_autorefresh(interval=2000, key="refresh")
