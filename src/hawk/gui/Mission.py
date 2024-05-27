# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, cast

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from hawk.gui.elements import ABOUT_TEXT, Mission, page_header
from hawk.mission_config import load_config

if TYPE_CHECKING:
    import pandas as pd
    from streamlit.delta_generator import DeltaGenerator

page_header("Hawk Browser")

####
# If no mission has been selected direct the user to either select an existing
# mission or configure/create a new one.
banner = st.empty()
if st.session_state.mission is None:
    with banner.container():
        st.title("Welcome to the Hawk Browser")
        st.markdown(
            f"""\
{ABOUT_TEXT}
### No Hawk mission selected
Choose an existing mission from the "**Select Mission**" pulldown in the
sidebar, or go to "**Mission Config**" to create and configure a new Hawk
mission.
"""
        )
        if st.button("Configure a new mission"):
            st.switch_page("pages/1_Config.py")
        st.stop()
else:
    banner.title("Hawk Mission Results")
    # banner.write(st.session_state)


mission = st.session_state.mission

# classes, first one is expected to be the not-yet-labeled value
# followed by the various classes in the mission
CLASSES = ["?", "neg", "pos"]

data = mission.get_data()
# banner.write(data)


###
# save/clear label controls in sidebar
#
def update_labels(mission: Mission) -> None:
    """update labels to include pending labels"""

    def extract_label_from_state(index: int) -> int:
        class_ = st.session_state.get(index, "?")
        return CLASSES.index(class_) - 1

    data = mission.get_data()
    pending = data[data.unlabeled].index.to_series().map(extract_label_from_state)
    mission.save_new_labels(pending)


def clear_labels() -> None:
    """clear pending (uncommitted) label values"""
    data = st.session_state.mission.get_data()
    for index in data.unlabeled[data.unlabeled].index:
        st.session_state[index] = "?"


col1, col2 = st.sidebar.columns(2)
with col1:
    st.button("Submit Labels", on_click=update_labels, args=(mission,))
with col2:
    st.button("Clear Labels", on_click=clear_labels)

statistics = st.sidebar.empty()


def update_statistics(mission: Mission, data: pd.DataFrame) -> bool:
    # read scout stats from logs/mission-stats.json
    stats = mission.get_stats()
    time_elapsed = int(stats.get("home_time", 0))
    samples_inferenced = int(stats.get("processedObjects", 0))
    samples_total = int(stats.get("totalObjects", samples_inferenced or 1))
    model_version = int(stats.get("version", 0))
    last_update = stats.get("last_update", 0)

    if time_elapsed == 0:
        mission_state = "Starting"
    elif (time.time() - last_update) > 60:
        mission_state = "Finished"
    else:
        mission_state = "Running"

    # compute home stats from received and labeled samples
    class_counts = data.imageLabel.value_counts()
    total_labeled = class_counts.sum()
    negative_labeled = cast(int, class_counts.get(0) or 0)
    positive_labeled = total_labeled - negative_labeled
    positive_label_ratio = (
        int(100 * positive_labeled / total_labeled) if total_labeled else 0
    )

    samples_received = len(data)
    received_ratio = (
        int(100 * samples_received / samples_inferenced) if samples_inferenced else 0
    )

    with statistics.container():
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Model Version", model_version)
            st.metric("Samples Inferenced", samples_inferenced)
            st.metric("Samples Received", samples_received)
            st.metric("Received Sample Ratio", f"{received_ratio}%")
            st.metric("Mission Status", mission_state)
        with col2:
            st.metric("Positives Labeled", positive_labeled)
            st.metric("Negatives Labeled", negative_labeled)
            st.metric("Samples Labeled", total_labeled)
            st.metric("Positive Sample Ratio", f"{positive_label_ratio}%")
            st.metric("Elapsed Mission Time", f"{time_elapsed}s")
        st.progress(float(samples_inferenced / samples_total))

    if mission_state == "Finished" and total_labeled == samples_received:
        st.session_state.show_labeled = True

    # return True if mission is still active
    return mission_state != "Finished"


####
# to minimize flickering when rerunning the script we update the sidebar
# before we render results.
if update_statistics(mission, data):
    st_autorefresh(interval=2000, key="refresh")

# sync session_state to persisted (on-disk) labels
for label in data.loc[:, ["imageLabel"]].dropna().itertuples():
    st.session_state[label.Index] = CLASSES[label.imageLabel + 1]


####
# Here we generate an infinite sequence of columns to put stuff into.
if "columns" not in st.session_state:
    st.session_state.columns = 4
st.sidebar.slider("columns", min_value=1, max_value=8, key="columns")
st.sidebar.toggle("Show Labeled", key="show_labeled")


def columns(ncols: int) -> Iterator[DeltaGenerator]:
    """Generator function to create infinite list of columns"""
    while 1:
        yield from st.columns(ncols)


column = columns(st.session_state.columns)


def display_radar_images(data: pd.DataFrame) -> None:
    unlabeled = data["unlabeled"]
    if not st.session_state.show_labeled:
        data = data[unlabeled]
    for row in data.itertuples():
        index = cast(int, row.Index)
        unlabeled = row.unlabeled
        objectid = row.objectId
        base = objectid.split("/")[-1].split(".")[0] + "_left.jpg"
        image = Path(mission.image_dir, f"{index:06}.jpeg")
        stereo_image = Path(
            "/media/eric/Drive2/RADAR_DETECTION/train/stereo_left/", f"{base}"
        )  # stereo image for radar missions, if stereo image.exists(), etc.
        with next(column):  # make a 1x2 container?
            col1, col2 = st.columns(2)
            with col1:
                view_height = 800
                img_height = 500
                padding_top = (view_height - img_height) // 2
                padding_bottom = view_height - img_height - padding_top
                st.header("Stereo")
                st.markdown(
                    f"<div style='padding-top: {padding_top}px'></div>",
                    unsafe_allow_html=True,
                )
                st.image(str(stereo_image), use_column_width=True)
                st.markdown(
                    f"<div style='padding-bottom: {padding_bottom}px'></div>",
                    unsafe_allow_html=True,
                )

            # col1.image(str(stereo_image))
            with col2:
                st.header("RD Map")
                st.image(str(image))
            st.radio(
                "classification",
                key=index,
                options=CLASSES,
                disabled=not unlabeled,
                label_visibility="collapsed",
                horizontal=st.session_state.columns <= 4,
            )


def display_images(data: pd.DataFrame) -> None:
    unlabeled = data["unlabeled"]
    if not st.session_state.show_labeled:
        data = data[unlabeled]

    for idx, unlabeled in data["unlabeled"].items():
        index = cast(int, idx)
        image = Path(mission.image_dir, f"{index:06}.jpeg")
        with next(column): 
            st.image(str(image))
            st.radio(
                "classification",
                key=index,
                options=CLASSES,
                disabled=not unlabeled,
                label_visibility="collapsed",
                horizontal=st.session_state.columns <= 4,
            )


config_file = mission.log_file
config = load_config(config_file)
train_strategy = config["train_strategy"]["type"]
if train_strategy == "dnn_classifier_radar":
    display_radar_images(data)  # only for radar missions
else:
    display_images(data)  # RGB default function call

# rely on autorefresh
# st.stop()

# while update_statistics(mission, data):
#    data_rows = data.index.size
#    mission.mission_data.resync_unlabeled()
#    new_results = mission.get_data().iloc[data_rows:]
#    if not new_results.empty:
#        display_images(new_results)
#    time.sleep(1)
