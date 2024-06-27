# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from hawk.gui.elements import ABOUT_TEXT, Mission, page_header
from hawk.home.label_utils import BoundingBox, LabelSample
from hawk.mission_config import load_config

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

start = time.time()
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
Choose a mission from the "**Select Mission**" pulldown in the sidebar.
"""
        )
        # if st.button("Configure a new mission"):
        #     st.switch_page("pages/1_Config.py")
        st.stop()
else:
    banner.title("Hawk Mission Results")
    # banner.write(st.session_state)


mission = st.session_state.mission
mission.resync()

# classes, first one is expected to be the not-yet-labeled value, second is
# negative, followed by the various classes in the mission
mission_classes = ["pos"]
CLASSES = ["?", "neg"] + mission_classes

# banner.write(data)


###
# save/clear label controls in sidebar
#
def update_labels(mission: Mission) -> None:
    """update labels to include pending labels"""

    def extract_label_from_state(index: int) -> int:
        class_ = st.session_state.get(index, "?")
        return CLASSES.index(class_) - 2

    pending = []
    for result in mission.unlabeled:
        if result.objectId in mission.labeled:
            continue

        label = extract_label_from_state(result.index)
        if label == -2:  # unlabeled
            continue

        if label == -1:  # negative
            result.labels = []
        else:
            result.labels = [BoundingBox(label=label)]
        pending.append(result)
    mission.save_labeled(pending)


def clear_labels(mission: Mission) -> None:
    """clear pending (uncommitted) label values"""
    for result in mission.unlabeled:
        if result.objectId not in mission.labeled:
            st.session_state[result.index] = "?"


col1, col2 = st.sidebar.columns(2)
with col1:
    st.button("Submit Labels", on_click=update_labels, args=(mission,))
with col2:
    st.button("Clear Labels", on_click=clear_labels, args=(mission,))

statistics = st.sidebar.empty()


def update_statistics(mission: Mission) -> bool:
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
    total_labeled = len(mission.labeled)
    negative_labeled = sum(1 for labels in mission.labeled.values() if not labels)
    positive_labeled = total_labeled - negative_labeled
    positive_label_ratio = (
        int(100 * positive_labeled / total_labeled) if total_labeled else 0
    )

    samples_received = len(mission.unlabeled)
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
if update_statistics(mission):
    st_autorefresh(interval=2000, key="refresh")

# sync session_state to persisted (on-disk) labels
for result in mission.unlabeled:
    if result.objectId not in mission.labeled:
        continue

    labels = mission.labeled[result.objectId]
    label = -1 if not labels else 0
    st.session_state[result.index] = CLASSES[label + 2]


####
# Here we generate an infinite sequence of columns to put stuff into.
if "columns" not in st.session_state:
    st.session_state["columns"] = 4
if "rows" not in st.session_state:
    st.session_state["rows"] = 2

st.sidebar.slider("columns", min_value=1, max_value=8, key="columns")
st.sidebar.slider("rows", min_value=1, max_value=8, key="rows")
st.sidebar.toggle("Show Labeled", key="show_labeled")


def columns(ncols: int) -> Iterator[DeltaGenerator]:
    """Generator function to create infinite list of columns"""
    while 1:
        yield from st.columns(ncols)


column = columns(st.session_state.columns)


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


def display_radar_images(mission: Mission) -> None:
    exclude = mission.labeled if not st.session_state.show_labeled else set()
    results = [result for result in mission.unlabeled if result.objectId not in exclude]

    with paginate(results) as page:
        for result in page:
            base = Path(result.objectId).stem
            image = Path(mission.image_dir, f"{result.index:06}.jpeg")
            stereo_image = Path(
                "/media/eric/Drive2/RADAR_DETECTION/train/stereo_left/",
                f"{base}_left.jpg",
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
                    key=result.index,
                    options=CLASSES,
                    disabled=result.objectId in mission.labeled,
                    label_visibility="collapsed",
                    horizontal=st.session_state.columns <= 4,
                )


def display_images(mission: Mission) -> None:
    exclude = mission.labeled if not st.session_state.show_labeled else set()
    results = [result for result in mission.unlabeled if result.objectId not in exclude]

    with paginate(results) as page:
        for result in page:
            image = Path(mission.image_dir, f"{result.index:06}.jpeg")
            with next(column):
                st.image(str(image))
                st.radio(
                    "classification",
                    key=result.index,
                    options=CLASSES,
                    disabled=result.objectId in mission.labeled,
                    label_visibility="collapsed",
                    horizontal=st.session_state.columns <= 4,
                )


config_file = mission.log_file
config = load_config(config_file)
train_strategy = config["train_strategy"]["type"]
if train_strategy == "dnn_classifier_radar":
    display_radar_images(mission)  # only for radar missions
else:
    display_images(mission)  # RGB default function call

elapsed = time.time() - start
st.write(f"Time to render page {elapsed:.3f}s")
