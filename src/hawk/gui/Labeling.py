# SPDX-FileCopyrightText: 2024-2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import streamlit as st
from blinker import Signal
from streamlit_label_kit import detection as st_detection

from hawk.classes import ClassList, ClassName, class_label_to_int
from hawk.gui.elements import (
    Mission,
    columns,
    load_mission,
    mission_changed,
    mission_stats,
    paginate,
)
from hawk.home.label_utils import Detection, LabelSample

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

st.title("Hawk Mission Results")
# st.write(st.session_state)


def reset_state_cb(sender: Signal | None) -> None:
    """Reset any mission specific session_state variables when we switched to a
    different mission"""
    st.session_state.saves = {}


mission_changed.connect(reset_state_cb)

if "saves" not in st.session_state:
    reset_state_cb(None)

mission = load_mission()
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

# need to set a default value for the selectbox
if "display_filter" not in st.session_state:
    st.session_state["display_filter"] = "Unlabeled"

with st.sidebar:
    mission_state = mission.state()
    mission_active = mission_state in ["Starting", "Running", "Training"]

    @st.fragment(run_every="2s" if mission_active else None)
    def update_stats() -> None:
        mission.resync()
        mission_stats(mission, None)
        if mission_active and mission.state() == "Finished":
            st.rerun()

    update_stats()

    if (
        mission_state == "Finished"
        and st.session_state.get("display_filter") == "Unlabeled"
    ):
        st.session_state["display_filter"] = "Positives"


####
# to minimize flickering when rerunning the script we update the sidebar
# before we render results.
dialog_displayed = False


####
# Here we generate an infinite sequence of columns to put stuff into.
if "columns" not in st.session_state:
    st.session_state["columns"] = 4
if "rows" not in st.session_state:
    st.session_state["rows"] = 2

st.sidebar.slider("columns", min_value=1, max_value=8, key="columns")
st.sidebar.slider("rows", min_value=1, max_value=8, key="rows")
st.sidebar.segmented_control(
    "Filter", ["Unlabeled", "Positives", "All"], key="display_filter"
)


# To inject a new class into the classification/detection.
def reset_new_class() -> None:
    for key in st.session_state:
        if key.endswith("_cls"):
            del st.session_state[key]


new_class = st.sidebar.text_input("New Class", on_change=reset_new_class)
if new_class:
    class_list.add(ClassName(sys.intern(new_class)))


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
    assert result.objectId is not None
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
    elif key in st.session_state.saves:
        del st.session_state.saves[result.index]

    if key != default_key:
        st.session_state[default_key] = classification


@st.dialog("Image Viewer", width="large")
def image_classifier_popup(mission: Mission, sample: LabelSample) -> None:
    image = mission.image_path(sample)
    st.image(str(image), use_container_width=True)

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
    assert sample.objectId is not None
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


def display_radar_images(mission: Mission, column: Iterator[DeltaGenerator]) -> None:
    exclude = mission.labeled
    display_filter = st.session_state.display_filter
    results = [
        result
        for result in mission.unlabeled
        if display_filter == "All" or result.objectId not in exclude
    ]

    results_per_page = st.session_state.rows * st.session_state.columns
    with paginate(results, results_per_page=results_per_page) as page:
        for result in page:
            assert result.objectId is not None
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
                    st.image(str(stereo_image), use_container_width=True)
                    st.markdown(
                        f"<div style='padding-bottom: {padding_bottom}px'></div>",
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.header("RD Map")
                    st.image(str(image))

                classification_pulldown(mission, result)


def display_images(
    mission: Mission, column: Iterator[DeltaGenerator], mark_negative: bool
) -> None:
    if st.session_state.display_filter == "All":
        results = mission.unlabeled
    elif st.session_state.display_filter == "Positives":
        results = [
            result
            for result in mission.unlabeled
            if result.objectId in mission.labeled
            and mission.labeled[result.objectId].detections
        ]
    elif st.session_state.display_filter == "Unlabeled":
        results = [
            result
            for result in mission.unlabeled
            if result.objectId not in mission.labeled
        ]

    results_per_page = st.session_state.rows * st.session_state.columns
    with paginate(results, results_per_page=results_per_page) as page:
        for result in page:
            with next(column):
                if result.is_classification:
                    key = f"{result.index}_cls"
                    if mark_negative and st.session_state.get(key) is None:
                        st.session_state[key] = "negative"

                    classification_ui(mission, result)
                else:
                    if (
                        mark_negative
                        and st.session_state.saves.get(result.index) is None
                    ):
                        st.session_state.saves[result.index] = []
                    detection_ui(mission, result)


@st.fragment(run_every="2s" if mission_active and not dialog_displayed else None)
def display_results() -> None:
    start = time.time()

    mission.blinkenlights()

    mark_negative = st.button("Default to Negative") if mission_active else False
    column = columns(st.session_state.columns)

    train_strategy = mission.config["train_strategy"]["type"]
    if train_strategy == "dnn_classifier_radar":
        display_radar_images(mission, column)  # only for radar missions
    else:
        display_images(mission, column, mark_negative)  # RGB default function call

    elapsed = time.time() - start
    st.caption(f"---\nTime to render fragment {elapsed:.3f}s")


display_results()
