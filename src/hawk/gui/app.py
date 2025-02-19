# SPDX-FileCopyrightText: 2024-2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import time

import streamlit as st

from hawk.gui.elements import ABOUT_TEXT, Mission, mission_changed, welcome_page

start = time.time()

st.set_page_config(
    page_title="Hawk Browser",
    page_icon=":material/thumbs_up_down:",
    menu_items={
        "Report a bug": "https://github.com/cmusatyalab/hawk/issues",
        "About": ABOUT_TEXT,
    },
    layout="wide",
)


def select_mission_cb() -> None:
    mission_changed.send()


# create "Select Mission" pulldown
st.sidebar.selectbox(
    "Select Mission",
    [None, *Mission.missions()],
    key="mission_name",
    on_change=select_mission_cb,
)


if st.session_state.get("mission_name") is None:
    pages = [welcome_page]
else:
    pages = [
        st.Page("Config.py", title="Configuration"),
        st.Page("Labeling.py", title="Labeling"),
        st.Page("Clustering.py", title="Clustering"),
        st.Page("Stats.py", title="Mission Stats"),
        # st.Page("Upload.py", title="Augment Training Data"),
    ]


app = st.navigation(pages)
app.run()

elapsed = time.time() - start
st.caption(f"---\nTime to render page {elapsed:.3f}s")
