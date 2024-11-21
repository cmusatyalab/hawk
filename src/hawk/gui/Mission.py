# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_label_kit import detection as st_detection

from hawk.classes import ClassList, ClassName, class_label_to_int
from hawk.gui.elements import ABOUT_TEXT, Mission, page_header, paginate
from hawk.home.label_utils import Detection, LabelSample

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

start = time.time()
mission = page_header("Hawk Browser")

####
# If no mission has been selected direct the user to either select an existing
# mission or configure/create a new one.
banner = st.empty()

if mission is None:
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

if "saves" not in st.session_state:
    st.session_state.saves = {}

mission.resync()

# list of positive classes in the mission
class_list = ClassList()
class_list.extend(mission.classes)
inprogress_classes = {
    cls
    for bboxes in st.session_state.saves.values()
    for det in bboxes
    for cls in det.scores
}
class_list.extend(inprogress_classes)

# banner.write(data)


###
# save/clear label controls in sidebar
#
def update_labels(mission: Mission) -> None:
    """update labels to include pending labels"""

    pending = []
    for sample in mission.unlabeled:
        if sample.objectId in mission.labeled:
            continue

        save = st.session_state.saves.get(sample.index)
        if save is not None:
            result = sample.replace(save)
            pending.append(result)

    mission.save_labeled(pending)
    st.session_state.saves = {}


def clear_labels() -> None:
    """clear pending (uncommitted) label values"""
    st.session_state.saves = {}


col1, col2 = st.sidebar.columns(2)
with col1:
    st.button("Submit Labels", on_click=update_labels, args=(mission,))
with col2:
    st.button("Clear Labels", on_click=clear_labels)

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
    negative_labeled = sum(
        1 for label in mission.labeled.values() if not label.detections
    )
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
mission_active = update_statistics(mission)
dialog_displayed = False


####
# Here we generate an infinite sequence of columns to put stuff into.
if "columns" not in st.session_state:
    st.session_state["columns"] = 4
if "rows" not in st.session_state:
    st.session_state["rows"] = 2

st.sidebar.slider("columns", min_value=1, max_value=8, key="columns")
st.sidebar.slider("rows", min_value=1, max_value=8, key="rows")
st.sidebar.toggle("Show Labeled", key="show_labeled")


# To inject a new class into the classification/detection.
def reset_new_class() -> None:
    for key in st.session_state:
        if key.endswith("_cls"):
            del st.session_state[key]


new_class = st.sidebar.text_input("New Class", on_change=reset_new_class)
if new_class:
    class_list.add(ClassName(sys.intern(new_class)))


def columns(ncols: int) -> Iterator[DeltaGenerator]:
    """Generator function to create infinite list of columns"""
    while 1:
        yield from st.columns(ncols)


column = columns(st.session_state.columns)


def classification_pulldown(
    mission: Mission, result: LabelSample, key: str | None = None
) -> None:
    scores = result.detections[0].scores if result.detections else {}
    options = ["negative"] + [
        f"{cls} ({scores.get(cls, 0):.02f})" for cls in class_list.positive
    ]

    default_key = f"{result.index}_cls"
    key = key or default_key

    # if we are initializing a new selectbox with no previously saved state
    labeled_result = mission.labeled.get(result.objectId)
    if key not in st.session_state:
        # find previously saved or in-progress detections
        if labeled_result is not None:
            detections = labeled_result.detections
        else:
            detections = st.session_state.saves.get(result.index)

        # if we found detections, pre-select the given option
        if detections is not None:
            class_index = 0
            if detections:
                class_name = detections[0].top_class()
                try:
                    class_label = class_list.index(class_name)
                    class_index = class_label_to_int(class_label)
                except ValueError:
                    # we should have recognized the class name.
                    # fall back to 'negative' class 0.
                    pass

            st.session_state[key] = options[class_index]

    classification = st.selectbox(
        "classification",
        options=options,
        key=key,
        index=None,
        placeholder="Select class...",
        disabled=labeled_result is not None,
        label_visibility="collapsed",
    )

    if classification is not None:
        if classification != "negative":
            name = classification.rsplit(" ", 1)[0]
            class_name = ClassName(sys.intern(name))
            st.session_state.saves[result.index] = [Detection(scores={class_name: 1.0})]
        else:
            st.session_state.saves[result.index] = []

        if key != default_key:
            st.session_state[default_key] = classification


@st.dialog("Image Viewer", width="large")
def image_classifier_popup(mission: Mission, sample: LabelSample) -> None:
    image = mission.image_path(sample)
    st.image(str(image), use_column_width=True)

    # with st.expander("See more"):
    #     st.image(str(image))

    classification_pulldown(mission, sample, key=f"{sample.index}_cls_popup")
    if st.button("Ok"):
        st.rerun()


def classification_ui(mission: Mission, sample: LabelSample) -> None:
    image = mission.image_path(sample)
    st.image(str(image))

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("View", key=f"{sample.index}_view"):
            global dialog_displayed
            dialog_displayed = True
            image_classifier_popup(mission, sample)
    with col2:
        classification_pulldown(mission, sample)


@st.dialog("Annotation Editor", width="large")
def annotation_editor_popup(mission: Mission, sample: LabelSample) -> None:
    image = mission.image_path(sample)
    out = st_detection(
        image_path=str(image),
        label_list=class_list.positive,
        bbox_format="REL_CXYWH",
        image_height=512,
        image_width=512,
        ui_size="large",
        class_select_position="bottom",
        component_alignment="center",
        bbox_show_label=True,
        # bbox_show_info=True,
        read_only=sample.objectId in mission.labeled,
        key=f"{sample.index}_editor",
        **st.session_state.editstate,
    )
    # with st.expander("See more"):
    #     st.image(str(image))
    if st.button("Ok"):
        if out is not None and out["key"] != 0:
            st.session_state.saves[sample.index] = [
                Detection.from_labelkit(bbox, class_list) for bbox in out["bbox"]
            ]
        st.rerun()


def detection_ui(mission: Mission, sample: LabelSample) -> None:
    labeled_result = mission.labeled.get(sample.objectId)
    inprogress_bboxes = st.session_state.saves.get(sample.index)

    # state is previously saved, in progress, or a new estimate from inference
    if labeled_result is not None:
        sample = sample.replace(labeled_result.detections)
    elif inprogress_bboxes is not None:
        sample = sample.replace(inprogress_bboxes)

    labelkit_args = sample.to_labelkit_args(class_list)

    # draw image with bounding boxes
    image = mission.image_path(sample)
    st_detection(
        image_path=str(image),
        label_list=class_list.positive,
        bbox_format="REL_CXYWH",
        ui_size="small",
        class_select_position="none",
        bbox_show_label=True,
        read_only=True,
        **labelkit_args,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Edit", key=f"{sample.index}_edit"):
            global dialog_displayed
            dialog_displayed = True
            st.session_state.editstate = labelkit_args
            annotation_editor_popup(mission, sample)

    with col2:
        new = labeled_result is None and inprogress_bboxes is None

        if not new:
            st.session_state[f"{sample.index}_fb"] = bool(sample.detections)

        feedback = st.feedback("thumbs", key=f"{sample.index}_fb")
        if new and feedback is not None:
            if not feedback:
                st.session_state.saves[sample.index] = []
                if sample.detections:
                    st.rerun()
            elif new:
                st.session_state.saves[sample.index] = sample.detections


def display_radar_images(mission: Mission) -> None:
    exclude = mission.labeled if not st.session_state.show_labeled else set()
    results = [result for result in mission.unlabeled if result.objectId not in exclude]

    with paginate(results) as page:
        for result in page:
            base = Path(result.objectId).stem
            stereo_base = base.split("_", 1)[0]

            image = mission.image_path(result)
            stereo_image = Path(
                "/media/eric/Drive2/RADAR_DETECTION/train/stereo_left/",
                f"{stereo_base}_left.jpg",
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

                with col2:
                    st.header("RD Map")
                    st.image(str(image))

                classification_pulldown(mission, result)


def display_images(mission: Mission) -> None:
    exclude = mission.labeled if not st.session_state.show_labeled else set()
    results = [result for result in mission.unlabeled if result.objectId not in exclude]

    with paginate(results) as page:
        for result in page:
            with next(column):
                if result.is_classification:
                    classification_ui(mission, result)
                else:
                    detection_ui(mission, result)


train_strategy = mission.config["train_strategy"]["type"]
if train_strategy == "dnn_classifier_radar":
    display_radar_images(mission)  # only for radar missions
else:
    display_images(mission)  # RGB default function call


###
# Display a by-class breakdown of labeled samples
counts: dict[int, Counter[ClassName]] = defaultdict(Counter)
for label in mission.labeled.values():
    counts[label.model_version].update(label.class_counts())
labeled_by_class = pd.DataFrame(counts)

col1, col2, *_ = st.columns(10)
detailed = col1.toggle("=")
as_percentage = col2.toggle("%")
if not labeled_by_class.empty:
    if not detailed:
        labeled_by_class = pd.DataFrame(labeled_by_class.T.sum())
    if as_percentage:
        labeled_by_class /= labeled_by_class.sum()
    elif "negative" in labeled_by_class.index:
        labeled_by_class.loc["negative"] *= -1
    st.bar_chart(labeled_by_class.T, horizontal=True, y_label="model version")
    # labeled_by_class = pd.DataFrame(labeled_by_class.T.sum())
    # st.bar_chart(labeled_by_class.T, horizontal=True)


elapsed = time.time() - start
st.write(f"Time to render page {elapsed:.3f}s")

if mission_active and not dialog_displayed:
    st_autorefresh(interval=2000, key="refresh")
