# SPDX-FileCopyrightText: 2024-2026 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import time

import streamlit as st

from hawk.gui.elements import Mission, empty_state, mission_changed, state_badge
from hawk.gui.mission_control import clone_mission

ABOUT_TEXT = """\
Hawk is a live learning system that leverages distributed machine learning,
human domain expertise, and edge computing to detect the presence of rare
objects and gather valuable training samples under austere and degraded
conditions.
"""


def open_mission_cb(mission_name: str) -> None:
    st.session_state["mission_name"] = mission_name
    mission_changed.send()
    mission = Mission.load(mission_name)
    st.session_state["_goto"] = (
        "setup" if mission.state() == "Not Started" else "labeling"
    )


def create_mission_cb(template_name: str) -> None:
    mission = Mission.load(f"_{template_name}")
    clone_mission(mission)
    mission_changed.send()
    st.session_state["_goto"] = "setup"


def mission_card(mission: Mission) -> None:
    current = st.session_state.get("mission_name") == mission.name
    modified = mission.mission_dir.stat().st_mtime

    with st.container(
        border=True,
        horizontal=True,
        vertical_alignment="center",
        gap="medium",
    ):
        with st.container(gap="xsmall"):
            with st.container(horizontal=True, gap="small"):
                st.markdown(f"**{mission.name}**")
                if current:
                    st.badge("Open", icon=":material/visibility:", color="blue")
            with st.container(horizontal=True, gap="small"):
                state_badge(mission.state())
                st.caption(
                    f"{len(mission.config.deploy.scouts)} scouts · updated "
                    f"{time.strftime('%b %d, %H:%M', time.localtime(modified))}",
                )
        st.button(
            "Open",
            key=f"open_{mission.name}",
            icon=":material/arrow_forward:",
            icon_position="right",
            type="secondary" if current else "primary",
            on_click=open_mission_cb,
            args=(mission.name,),
        )


def template_card(template: Mission) -> None:
    with st.container(border=True, gap="small"):
        st.markdown(f"**{template.name.lstrip('_')}**")
        if template.description:
            st.markdown(template.description)
        else:
            st.caption("No description")
        train_type = template.config.get("train_strategy", {}).get("type", "-")
        st.caption(f":material/model_training: {train_type}")
        st.button(
            "Create mission",
            key=f"create_{template.name}",
            icon=":material/add:",
            on_click=create_mission_cb,
            args=(template.name.lstrip("_"),),
            width="stretch",
        )


def about_hawk() -> None:
    st.title("Missions", anchor=False)
    st.caption(ABOUT_TEXT)

    all_missions = [Mission.load(name) for name in Mission.missions()]
    missions = sorted(
        (mission for mission in all_missions if not mission.is_template),
        key=lambda mission: mission.mission_dir.stat().st_mtime,
        reverse=True,
    )
    templates = [mission for mission in all_missions if mission.is_template]

    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        with st.container(horizontal=True, vertical_alignment="bottom"):
            st.subheader("Recent missions", anchor=False)
            search = st.text_input(
                "Search missions",
                placeholder="Search…",
                icon=":material/search:",
                label_visibility="collapsed",
                width=220,
            )
        if search:
            missions = [m for m in missions if search.lower() in m.name.lower()]

        if not missions:
            empty_state(
                "No missions yet" if not search else "No matching missions",
                "Create a new mission from one of the templates."
                if not search
                else "Try a different search term.",
                icon=":material/travel_explore:",
            )
        for mission in missions:
            mission_card(mission)

    with col2:
        st.subheader("Start from a template", anchor=False)
        if not templates:
            empty_state(
                "No templates found",
                "Template missions are mission directories whose name starts "
                "with an underscore, e.g. `_my-template`.",
                icon=":material/content_copy:",
            )
        for template in templates:
            template_card(template)


welcome_page = st.Page(
    about_hawk,
    title="Missions",
    icon=":material/home:",
    url_path="missions",
    default=True,
)
