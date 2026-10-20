# SPDX-FileCopyrightText: 2024-2026 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import streamlit as st

from hawk.gui.elements import (
    Mission,
    columns,
    empty_state,
    load_mission,
    page_header,
    paginate,
)
from hawk.home.label_utils import LabelSample, read_jsonl

ALL_SCOUTS = "All scouts"

mission = load_mission()

if "cluster_columns" not in st.session_state:
    st.session_state["cluster_columns"] = 6
if "cluster_rows" not in st.session_state:
    st.session_state["cluster_rows"] = 4

actions = page_header("Novel clusters", mission)
with actions, st.popover("Layout", icon=":material/grid_view:"):
    st.slider("Columns", min_value=2, max_value=12, key="cluster_columns")
    st.slider("Rows", min_value=1, max_value=12, key="cluster_rows")

st.caption(
    "Representative samples from clusters that the scouts could not match to a "
    "known class. They may point at a new class worth labeling.",
)


@st.dialog("Novel sample", width="large", icon=":material/bubble_chart:")
def image_zoom_popup(mission: Mission, sample: LabelSample) -> None:
    image = sample.content(mission.mission_dir / "novel", index=0)
    col1, col2 = st.columns([3, 2], gap="medium")
    with col1:
        st.image(str(image), width="stretch")
    with col2:
        st.badge(f"Scout {sample.scoutIndex}", icon=":material/satellite_alt:")
        st.caption(
            f"`{sample.objectId.serialize_oid() if sample.objectId else ''}`",
        )
        if st.button("Close", type="primary", width="stretch"):
            st.rerun()


def display_cluster(mission: Mission, sample: LabelSample) -> None:
    image = sample.content(mission.mission_dir / "novel", index=0)
    with st.container(border=True, gap="xsmall"):
        st.image(str(image), width="stretch")
        with st.container(
            horizontal=True,
            vertical_alignment="center",
            horizontal_alignment="distribute",
        ):
            st.caption(f"Scout {sample.scoutIndex}")
            if st.button(
                "",
                icon=":material/open_in_full:",
                type="tertiary",
                key=f"{sample.index}_view",
                help="Open larger view",
            ):
                image_zoom_popup(mission, sample)


@st.fragment(run_every="10s" if mission.is_active else None)
def display_images() -> None:
    clusters = list(read_jsonl(mission.mission_dir / "novel.jsonl"))
    if not clusters:
        empty_state(
            "No novel clusters yet" if mission.is_active else "No novel clusters",
            "Scouts report clusters of unfamiliar samples while the mission runs."
            if mission.is_active
            else "This mission did not report any novel clusters.",
            icon=":material/bubble_chart:",
        )
        return

    scouts = sorted({sample.scoutIndex for sample in clusters})
    if len(scouts) > 1:
        scout = st.segmented_control(
            "Scout",
            [ALL_SCOUTS, *scouts],
            default=ALL_SCOUTS,
            format_func=lambda s: s if s == ALL_SCOUTS else f"Scout {s}",
            required=True,
            label_visibility="collapsed",
            key="cluster_scout",
        )
        if scout != ALL_SCOUTS:
            clusters = [sample for sample in clusters if sample.scoutIndex == scout]

    column = columns(st.session_state.cluster_columns)
    results_per_page = st.session_state.cluster_columns * st.session_state.cluster_rows
    with paginate(clusters, results_per_page=results_per_page) as page:
        for sample in page:
            with next(column):
                display_cluster(mission, sample)


display_images()
