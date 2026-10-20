# SPDX-FileCopyrightText: 2024-2026 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import streamlit as st

from hawk.gui.elements import (
    Mission,
    mission_changed,
    mission_stats,
    state_badge,
)
from hawk.gui.Welcome import ABOUT_TEXT, welcome_page

st.set_page_config(
    page_title="Hawk Browser",
    page_icon=":material/thumbs_up_down:",
    menu_items={
        "Report a bug": "https://github.com/cmusatyalab/hawk/issues",
        "About": ABOUT_TEXT,
    },
    layout="wide",
)

PAGES = {
    "setup": st.Page(
        "Config.py",
        title="Setup",
        icon=":material/tune:",
        url_path="setup",
    ),
    "bootstrap": st.Page(
        "Bootstrap.py",
        title="Bootstrap",
        icon=":material/photo_library:",
        url_path="bootstrap",
    ),
    "labeling": st.Page(
        "Labeling.py",
        title="Labeling",
        icon=":material/label:",
        url_path="labeling",
    ),
    "clusters": st.Page(
        "Clustering.py",
        title="Clusters",
        icon=":material/bubble_chart:",
        url_path="clusters",
    ),
    "stats": st.Page(
        "Stats.py",
        title="Statistics",
        icon=":material/monitoring:",
        url_path="stats",
    ),
}

missions = [mission for mission in Mission.missions() if not mission.startswith("_")]

# On a fresh page load the mission comes from the URL, we need it early to
# decide which pages are available.
if "mission_name" not in st.session_state:
    st.session_state["mission_name"] = st.query_params.get("mission_name")

# a mission was removed (or renamed) behind our back
if st.session_state.get("mission_name") not in [None, *missions]:
    st.session_state["mission_name"] = None

if st.session_state["mission_name"] is None:
    app = st.navigation([welcome_page], position="hidden")
else:
    app = st.navigation([welcome_page, *PAGES.values()], position="top")

    goto = st.session_state.pop("_goto", None)
    if goto in PAGES:
        st.switch_page(
            PAGES[goto],
            query_params={"mission_name": st.session_state["mission_name"]},
        )


def select_mission_cb() -> None:
    mission_changed.send()
    mission_name = st.session_state.get("mission_name")
    if mission_name is not None:
        mission = Mission.load(mission_name)
        st.session_state["_goto"] = (
            "setup" if mission.state() == "Not Started" else "labeling"
        )


def mission_status(mission_name: str, was_active: bool) -> None:
    """Live mission status in the sidebar."""
    mission = Mission.load(mission_name)
    mission.resync()
    state = mission.state()
    if was_active and not mission.is_active:
        # mission finished, rerun the whole app to stop the periodic refresh
        st.rerun()

    with st.container(border=True, gap="small"):
        with st.container(horizontal=True, horizontal_alignment="distribute"):
            st.caption("**Status**")
            state_badge(state)
        if state == "Not Started":
            st.caption("Deploy the scouts and start the mission from Setup.")
        else:
            mission.blinkenlights(compact=True)
            mission_stats(mission, None, compact=True)


with st.sidebar:
    st.selectbox(
        "Mission",
        missions,
        index=None,
        key="mission_name",
        placeholder="Choose a mission…",
        on_change=select_mission_cb,
    )
    mission_name = st.session_state.get("mission_name")

    # keep the selected mission in the URL so views can be bookmarked/shared
    if mission_name is None:
        st.query_params.pop("mission_name", None)
    elif st.query_params.get("mission_name") != mission_name:
        st.query_params["mission_name"] = mission_name
    if mission_name is not None:
        active = Mission.load(mission_name).is_active
        st.fragment(mission_status, run_every="2s" if active else None)(
            mission_name,
            active,
        )


app.run()
