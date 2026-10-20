# SPDX-FileCopyrightText: 2024-2026 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Iterator

import streamlit as st
from streamlit_label_kit import detection as st_detection

from hawk import Detection
from hawk.classes import ClassList, ClassName, class_label_to_int
from hawk.gui.elements import (
    Mission,
    columns,
    empty_state,
    load_mission,
    mission_changed,
    page_header,
    paginate,
)

if TYPE_CHECKING:
    from blinker import Signal
    from streamlit.delta_generator import DeltaGenerator

    from hawk.home.label_utils import LabelSample


def reset_state_cb(sender: Signal | None) -> None:
    """Reset any mission specific session_state variables when we switched to a
    different mission.
    """
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
    det.class_name for bboxes in st.session_state.saves.values() for det in bboxes
}
class_list.extend(inprogress_classes)


###
# save/clear label controls in sidebar
#
def update_labels(mission: Mission) -> None:
    """Update labels to include pending labels."""
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
    st.session_state["_submitted"] = len(pending)


def clear_labels() -> None:
    """Clear pending (uncommitted) label values."""
    st.session_state.saves = {}


mission_state = mission.state()
mission_active = mission_state in ["Starting", "Running", "Training"]

# show unlabeled samples by default, or the positives once a mission finished
if "display_filter" not in st.session_state:
    st.session_state["display_filter"] = (
        "Positives" if mission_state == "Finished" else "Unlabeled"
    )

if "columns" not in st.session_state:
    st.session_state["columns"] = 4
if "rows" not in st.session_state:
    st.session_state["rows"] = 2

####
# to minimize flickering when rerunning the script we render the controls
# before we render results.
dialog_displayed = False


# To inject a new class into the classification/detection.
def reset_new_class() -> None:
    for key in st.session_state:
        if isinstance(key, str) and key.endswith("_cls"):
            del st.session_state[key]


submitted = st.session_state.pop("_submitted", None)
if submitted is not None:
    st.toast(
        f"Submitted {submitted} label{'s' if submitted != 1 else ''}",
        icon=":material/check_circle:",
    )

actions = page_header("Labeling", mission)
with actions:
    with st.popover("Classes", icon=":material/category:"):
        st.caption("Positive classes in this mission")
        with st.container(horizontal=True, gap="small"):
            for cls in class_list.positive:
                st.badge(str(cls), color="blue")
        new_class = st.text_input(
            "Add a class",
            placeholder="e.g. truck",
            on_change=reset_new_class,
            help="Adds a new class to the pulldowns, it becomes part of the "
            "mission once a sample is labeled with it.",
        )
    with st.popover("Layout", icon=":material/grid_view:"):
        st.slider("Columns", min_value=1, max_value=8, key="columns")
        st.slider("Rows", min_value=1, max_value=8, key="rows")

if new_class:
    class_list.add(ClassName(sys.intern(new_class)))

n_unlabeled = sum(1 for r in mission.unlabeled if r.objectId not in mission.labeled)
n_positives = sum(
    1
    for r in mission.unlabeled
    if r.objectId in mission.labeled and mission.labeled[r.objectId].detections
)
filter_counts = {
    "Unlabeled": n_unlabeled,
    "Positives": n_positives,
    "All": len(mission.unlabeled),
}
st.segmented_control(
    "Show",
    list(filter_counts),
    key="display_filter",
    format_func=lambda option: f"{option} · {filter_counts[option]:,}",
    required=True,
    label_visibility="collapsed",
)


def classification_pulldown(
    mission: Mission,
    result: LabelSample,
    key: str | None = None,
) -> None:
    scores = {
        detection.class_name: detection.confidence for detection in result.detections
    }
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
            detections = Detection.sort_detections(detections)
            class_index = 0
            if detections:
                class_name = detections[0].class_name
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
        placeholder="Select class…",
        disabled=labeled_result is not None,
        label_visibility="collapsed",
    )

    if classification is not None:
        if classification != "negative":
            name = classification.rsplit(" ", 1)[0]
            class_name = ClassName(sys.intern(name))
            st.session_state.saves[result.index] = [Detection(class_name=class_name)]
        else:
            st.session_state.saves[result.index] = []
    elif result.index in st.session_state.saves:
        del st.session_state.saves[result.index]

    if key != default_key:
        st.session_state[default_key] = classification


def label_status(mission: Mission, sample: LabelSample) -> None:
    """Small badge showing whether a sample is labeled, pending or new."""
    assert sample.objectId is not None
    labeled_result = mission.labeled.get(sample.objectId)
    pending = st.session_state.saves.get(sample.index)

    if labeled_result is not None:
        classes = sorted({str(d.class_name) for d in labeled_result.detections})
        if classes:
            st.badge(", ".join(classes), icon=":material/check:", color="green")
        else:
            st.badge("negative", icon=":material/check:", color="gray")
    elif pending is not None:
        st.badge("pending", icon=":material/edit:", color="blue")
    else:
        st.badge(
            f"{sample.max_score:.2f}",
            icon=":material/insights:",
            color="gray",
            help="Highest confidence score from the scout's model",
        )


@st.dialog("Sample", width="large", icon=":material/image:")
def image_classifier_popup(mission: Mission, sample: LabelSample) -> None:
    images = [
        str(mission.image_path(sample, index=index))
        for index in range(len(sample.oracle_items))
    ]

    col1, col2 = st.columns([3, 2], gap="medium")
    with col1:
        st.image(images, width="stretch")
    with col2, st.container(gap="small"):
        label_status(mission, sample)
        st.caption(
            f"Scout {sample.scoutIndex} · model v{sample.model_version}  \n"
            f"`{sample.objectId.serialize_oid() if sample.objectId else ''}`",
        )
        for detection in Detection.sort_detections(sample.detections):
            st.progress(
                detection.confidence,
                text=f"{detection.class_name} · {detection.confidence:.2f}",
            )
        classification_pulldown(mission, sample, key=f"{sample.index}_cls_popup")
        if st.button("Done", type="primary", width="stretch"):
            st.rerun()


def classification_ui(mission: Mission, sample: LabelSample) -> None:
    image = mission.image_path(sample, index=0)
    st.image(str(image), width="stretch")

    with st.container(
        horizontal=True,
        vertical_alignment="center",
        horizontal_alignment="distribute",
        gap="small",
    ):
        label_status(mission, sample)
        if st.button(
            "",
            icon=":material/open_in_full:",
            type="tertiary",
            key=f"{sample.index}_view",
            help="Open larger view",
        ):
            global dialog_displayed
            dialog_displayed = True
            image_classifier_popup(mission, sample)
    classification_pulldown(mission, sample)


@st.dialog("Annotation editor", width="large", icon=":material/edit_square:")
def annotation_editor_popup(mission: Mission, sample: LabelSample) -> None:
    image = mission.image_path(sample, index=0)
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

    if len(sample.oracle_items) > 1:
        with st.expander("Additional views"):
            images = [
                str(mission.image_path(sample, index=index))
                for index in range(1, len(sample.oracle_items))
            ]
            st.image(images)

    with st.container(horizontal=True, horizontal_alignment="right"):
        done = st.button("Done", type="primary")
    if done:
        if out is not None and out["key"] != 0:
            st.session_state.saves[sample.index] = [
                Detection.from_labelkit(bbox, class_list) for bbox in out["bbox"]
            ]
        st.rerun()


def detection_ui(mission: Mission, sample: LabelSample) -> None:
    assert sample.objectId is not None
    labeled_result = mission.labeled.get(sample.objectId)
    inprogress_bboxes: list[Detection] = st.session_state.saves.get(sample.index)

    # state is previously saved, in progress, or a new estimate from inference
    if labeled_result is not None:
        sample = sample.replace(labeled_result.detections)
    elif inprogress_bboxes is not None:
        sample = sample.replace(inprogress_bboxes)

    labelkit_args = sample.to_labelkit_args(class_list)

    # draw image with bounding boxes
    image = mission.image_path(sample, index=0)
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

    with st.container(
        horizontal=True,
        vertical_alignment="center",
        horizontal_alignment="distribute",
        gap="small",
    ):
        label_status(mission, sample)
        new = labeled_result is None and inprogress_bboxes is None

        if not new:
            # st.feedback("thumbs") uses 0 for thumbs down, 1 for thumbs up
            st.session_state[f"{sample.index}_fb"] = int(bool(sample.detections))

        feedback = st.feedback("thumbs", key=f"{sample.index}_fb")
        if new and feedback is not None:
            if not feedback:
                st.session_state.saves[sample.index] = []
                if sample.detections:
                    st.rerun()
            elif new:
                st.session_state.saves[sample.index] = sample.detections

        if st.button(
            "",
            icon=":material/edit_square:",
            type="tertiary",
            key=f"{sample.index}_edit",
            help="Edit bounding boxes",
        ):
            global dialog_displayed
            dialog_displayed = True
            st.session_state.editstate = labelkit_args
            annotation_editor_popup(mission, sample)


def display_radar_images(mission: Mission, column: Iterator[DeltaGenerator]) -> None:
    exclude = mission.labeled
    display_filter = st.session_state.display_filter
    results = [
        result
        for result in mission.unlabeled
        if display_filter == "All" or result.objectId not in exclude
    ]
    if not results:
        no_results()
        return

    results_per_page = st.session_state.rows * st.session_state.columns
    with paginate(results, results_per_page=results_per_page) as page:
        for result in page:
            image = mission.image_path(result, index=0)

            # stereo image for radar missions, if stereo image.exists(), etc.
            stereo_image = mission.image_path(result, index=1)

            with next(column), st.container(border=True, gap="small"):
                col1, col2 = st.columns(2, vertical_alignment="center")
                with col1:
                    st.caption("Stereo")
                    st.image(str(stereo_image), width="stretch")
                with col2:
                    st.caption("RD map")
                    st.image(str(image), width="stretch")

                label_status(mission, result)
                classification_pulldown(mission, result)


def no_results() -> None:
    display_filter = st.session_state.display_filter
    if display_filter == "Unlabeled" and mission_active:
        empty_state(
            "All caught up",
            "Every received sample has been labeled. New samples show up here "
            "as soon as the scouts send them.",
            icon=":material/task_alt:",
        )
    elif display_filter == "Unlabeled":
        empty_state(
            "Nothing to label",
            "There are no unlabeled samples for this mission.",
            icon=":material/task_alt:",
        )
    elif display_filter == "Positives":
        empty_state(
            "No positives yet",
            "Samples you label with a positive class show up here.",
            icon=":material/search:",
        )
    else:
        empty_state(
            "No samples received yet",
            "Samples show up here once the scouts start sending results.",
            icon=":material/hourglass_empty:",
        )


def display_images(
    mission: Mission,
    column: Iterator[DeltaGenerator],
    mark_negative: bool,
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

    if not results:
        no_results()
        return

    results_per_page = st.session_state.rows * st.session_state.columns
    with paginate(results, results_per_page=results_per_page) as page:
        for result in page:
            with next(column), st.container(border=True, gap="small"):
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
    # labels were submitted from within this fragment, rerun the whole page to
    # update the counters outside of the fragment.
    if "_submitted" in st.session_state:
        st.rerun()

    pending = len(st.session_state.saves)

    with st.container(
        horizontal=True,
        vertical_alignment="center",
        horizontal_alignment="distribute",
    ):
        st.caption(
            f"**{pending}** pending label{'s' if pending != 1 else ''}"
            if pending
            else "Review the samples below, then submit your labels.",
        )
        with st.container(horizontal=True, width="content", gap="small"):
            mark_negative = (
                st.button(
                    "Mark rest as negative",
                    icon=":material/block:",
                    type="tertiary",
                    help="Label every sample on this page that does not have a "
                    "label yet as negative.",
                )
                if st.session_state.display_filter == "Unlabeled"
                else False
            )
            st.button(
                "Clear",
                icon=":material/undo:",
                on_click=clear_labels,
                disabled=not pending,
            )
            st.button(
                f"Submit {pending} label{'s' if pending != 1 else ''}"
                if pending
                else "Submit labels",
                icon=":material/send:",
                type="primary",
                on_click=update_labels,
                args=(mission,),
                disabled=not pending,
                shortcut="ctrl+enter",
            )

    column = columns(st.session_state.columns)

    train_strategy = mission.config["train_strategy"]["type"]
    if train_strategy == "dnn_classifier_radar":
        display_radar_images(mission, column)  # only for radar missions
    else:
        display_images(mission, column, mark_negative)  # RGB default function call


display_results()
