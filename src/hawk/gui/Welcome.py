# SPDX-FileCopyrightText: 2024-2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import streamlit as st

from hawk.gui.elements import Mission
from hawk.gui.mission_control import clone_mission

ABOUT_TEXT = """\
Hawk is a live learning system that leverages distributed machine learning,
human domain expertise, and edge computing to detect the presence of rare
objects and gather valuable training samples under austere and degraded
conditions.
"""


def create_mission_cb() -> None:
    template_name = st.session_state["template_name"]
    if template_name is not None:
        mission = Mission.load(f"_{template_name}")
        clone_mission(mission)


def about_hawk() -> None:
    st.title("Welcome to the Hawk Browser")
    st.markdown(
        f"""\
{ABOUT_TEXT}

### No Hawk mission selected

Choose a mission from the "**Select Mission**" pulldown in the sidebar.
Or choose a "**Template Mission**" to create a new mission from scratch.

### Template Mission
""",
    )
    col1, col2 = st.columns(2)
    template_name = col1.selectbox(
        "Templates",
        [mission[1:] for mission in Mission.missions() if mission.startswith("_")],
        key="template_name",
        label_visibility="collapsed",
    )
    col2.button("Create Mission", on_click=create_mission_cb)
    if template_name is not None:
        mission = Mission.load(f"_{template_name}")
        if mission.description:
            with st.container(border=True):
                st.markdown(mission.description)


welcome_page = st.Page(about_hawk)
