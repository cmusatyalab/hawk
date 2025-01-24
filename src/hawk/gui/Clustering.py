# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from pathlib import Path

import streamlit as st

from hawk.gui.elements import Mission, columns, load_mission, paginate
from hawk.home.label_utils import LabelSample, read_jsonl

st.title("Novel Cluster Explorer")
# st.write(st.session_state)

mission = load_mission()
clusters = list(read_jsonl(Path(mission.mission_dir, "novel.jsonl")))
# st.write(clusters)

dialog_displayed = False


####
# Hardcoded constants, should be configurable at some point
samples_per_cluster = 2
clusters_per_scout = 5
number_of_scouts = 8

display_columns = samples_per_cluster * clusters_per_scout
display_rows = number_of_scouts

####
# Here we generate an infinite sequence of columns to put stuff into.
column = columns(display_columns)


@st.dialog("Image Viewer", width="large")
def image_zoom_popup(mission: Mission, sample: LabelSample) -> None:
    image = sample.content(Path(mission.mission_dir) / "novel", ".jpeg")
    st.image(str(image), use_container_width=True)
    if st.button("Ok"):
        st.rerun()


def display_cluster(mission: Mission, sample: LabelSample) -> None:
    image = sample.content(Path(mission.mission_dir) / "novel", ".jpeg")
    st.image(str(image))

    if st.button("View", key=f"{sample.index}_view"):
        global dialog_displayed
        dialog_displayed = True
        image_zoom_popup(mission, sample)


def display_images(results: list[LabelSample]) -> None:
    results_per_page = display_columns * display_rows
    with paginate(results, results_per_page=results_per_page) as page:
        for result in page:
            with next(column):
                display_cluster(mission, result)


display_images(clusters)  # RGB default function call
