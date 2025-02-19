# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import shutil
import time
import zipfile
from pathlib import Path
from typing import Callable

import streamlit as st

from hawk.deploy_config import DeployConfig
from hawk.gui import deployment
from hawk.gui.elements import Mission

# Mission control commands that show up in multiple places
_CMD_CREATE = "Create new mission"
_CMD_START_MISSION = "Start mission"
_CMD_STOP_MISSION = "Stop mission"
_CMD_RESET = "Reset mission state"
_CMD_DELETE = "Destroy all mission state and configuration"
_CMD_CHECK_SCOUTS = "Check scouts"
_CMD_DEPLOY = "Deploy scouts"
_CMD_START_SCOUTS = "Start scouts"
_CMD_RESTART_SCOUTS = "Restart scouts"
_CMD_STOP_SCOUTS = "Stop scouts"

if st.session_state.get("deployed_state") is None:
    st.session_state.deployed_state = []


@st.dialog("Are you sure?")
def _confirm(
    callback: Callable[[Mission], bool], mission: Mission, prompt: str
) -> None:
    with st.form("Confirm", enter_to_submit=False):
        confirmed = st.form_submit_button(f"Yes, {prompt}")

    if confirmed and callback(mission):
        time.sleep(2)
        st.rerun()


@st.dialog("Executing command")
def _progress(callback: Callable[[DeployConfig], bool], mission: Mission) -> None:
    if callback(mission.config.deploy):
        time.sleep(2)
        st.rerun()


def archive_mission_state(mission: Mission) -> None:
    """Archive mission state."""

    mission_dir = Path(mission.mission_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    archive_path = mission_dir.joinpath(timestamp).with_suffix(".zip")

    with zipfile.ZipFile(
        archive_path, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive_paths = (
            ["mission_config.yml"]
            + mission.extra_config_files
            + [
                "logs",
                "traces",
                "unlabeled.jsonl",
                "labeled.jsonl",
                "images",
                "novel.jsonl",
                "novel",
                "feature_vectors",
            ]
        )
        for file in archive_paths:
            path = mission_dir / file
            if path.is_dir():
                # archive.mkdir(path)
                for sub_path in path.iterdir():
                    if sub_path.is_file():
                        arcname = sub_path.relative_to(mission_dir)
                        archive.write(sub_path, str(arcname))
            elif path.is_file():
                archive.write(path, file)


def clone_mission(mission: Mission) -> bool:
    """Create a new mission from an existing one."""

    mission_dir = Path(mission.mission_dir)
    mission_name = mission.config.get("mission-name", mission.name.lstrip("_"))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    new_mission_name = f"{mission_name}-{timestamp}"
    new_mission_dir = mission_dir.parent.joinpath(new_mission_name)
    with st.status(f"Creating Mission {new_mission_name}...", expanded=True) as status:
        st.write("Creating mission directory...")
        time.sleep(1)
        new_mission_dir.mkdir()
        st.write("Copying mission configuration...")
        for file in ["mission_config.yml"] + mission.extra_config_files:
            path = mission_dir / file
            if path.exists():
                st.write(f"Copying {file}...")
                new_path = new_mission_dir / file
                shutil.copy(path, new_path)
                time.sleep(1)
        status.update(
            label=f"Mission {new_mission_name} created", state="complete", expanded=True
        )
    st.session_state["mission_name"] = new_mission_name
    return True


def reset_mission(mission: Mission) -> bool:
    """Archive and reset mission state (labels/images/logs)."""

    mission_dir = Path(mission.mission_dir)
    with st.status("Resetting Mission state...", expanded=True) as status:
        st.write("Archiving mission state...")
        archive_mission_state(mission)

        st.write("Removing labeled/unlabeled data...")
        time.sleep(0.5)
        mission_dir.joinpath("unlabeled.jsonl").unlink(missing_ok=True)
        mission_dir.joinpath("labeled.jsonl").unlink(missing_ok=True)
        shutil.rmtree(mission_dir / "images", ignore_errors=True)
        st.write("Removing novel class examples...")
        time.sleep(0.5)
        mission_dir.joinpath("novel.jsonl").unlink(missing_ok=True)
        shutil.rmtree(mission_dir / "novel", ignore_errors=True)
        shutil.rmtree(mission_dir / "feature_vectors", ignore_errors=True)
        st.write("Removing logs...")
        time.sleep(0.5)
        shutil.rmtree(mission_dir / "traces", ignore_errors=True)
        shutil.rmtree(mission_dir / "logs", ignore_errors=True)
        status.update(label="Mission reset", state="complete", expanded=True)
    return True


def delete_mission(mission: Mission) -> bool:
    """Completely destroy all state and configuration."""

    mission_dir = Path(mission.mission_dir)
    with st.status("Deleting Mission...", expanded=True) as status:
        st.write("Removing labeled/unlabeled data...")
        time.sleep(1)
        st.write("Removing novel class examples...")
        time.sleep(1)
        st.write("Removing logs...")
        time.sleep(1)
        st.write("Removing archived state...")
        time.sleep(1)
        st.write("Removing bootstrap examples...")
        time.sleep(1)
        st.write("Removing mission config...")
        time.sleep(1)
        st.write("Removing mission directory...")
        time.sleep(1)
        shutil.rmtree(mission_dir, ignore_errors=True)
        status.update(label="Mission deleted", state="complete", expanded=True)
    st.session_state["mission_name"] = None
    return True


@st.dialog("Starting mission")
def start_home(mission: Mission) -> None:
    """Start the mission, if all scouts are deployed."""

    n_deployed = len(st.session_state.get("deployed_state", []))
    n_scouts = len(mission.config.deploy.scouts)
    if n_deployed != n_scouts:
        st.error("Not all scouts are deployed")

    mission_dir = Path(mission.mission_dir)
    with st.status("Starting Mission...", expanded=True) as status:
        st.write("Starting Hawk process...")
        deployment.start_home(mission_dir)
        time.sleep(1)
        status.update(label="Mission started", state="complete", expanded=True)
    time.sleep(2)
    st.rerun()


@st.dialog("Stopping mission")
def stop_home(mission: Mission) -> None:
    """Stop the mission and scouts."""

    deployment.stop_scouts(mission.config.deploy)

    mission_dir = Path(mission.mission_dir)
    with st.status("Stopping Mission...", expanded=True) as status:
        st.write("Stopping Hawk process...")
        deployment.stop_home(mission_dir)
        time.sleep(1)
        status.update(label="Mission terminated", state="complete", expanded=True)
    time.sleep(2)
    st.rerun()


def _action() -> None:
    action = st.session_state["controls"] or st.session_state["advanced_controls"]
    st.session_state["controls"] = None
    st.session_state["advanced_controls"] = None
    st.session_state["action"] = action


def _do_mission_control(mission: Mission) -> None:
    manage = st.session_state.action
    st.session_state.action = None
    if _CMD_CREATE in manage:
        _confirm(clone_mission, mission, "create a new mission from selected")
    elif _CMD_CHECK_SCOUTS in manage:
        _progress(deployment.check_scouts, mission)
    elif _CMD_DEPLOY in manage:
        _progress(deployment.deploy_scouts, mission)
    elif _CMD_START_SCOUTS in manage or _CMD_RESTART_SCOUTS in manage:
        _progress(deployment.restart_scouts, mission)
    elif _CMD_STOP_SCOUTS in manage:
        _progress(deployment.stop_scouts, mission)
    elif _CMD_START_MISSION in manage:
        start_home(mission)
    elif _CMD_STOP_MISSION in manage:
        stop_home(mission)
    elif _CMD_RESET in manage:
        _confirm(reset_mission, mission, "reset all mission state")
    elif _CMD_DELETE in manage:
        _confirm(delete_mission, mission, "delete all mission state and configuration")


def mission_controls(mission: Mission) -> None:
    if st.session_state.get("action"):
        _do_mission_control(mission)

    mission_state = mission.state()

    actions = [_CMD_CREATE]
    if not mission.name.startswith("_"):
        n_deployed = len(st.session_state.get("deployed_state", []))
        n_scouts = len(mission.config.deploy.scouts)
        if mission_state == "Not Started":
            actions.append(_CMD_DEPLOY)
            # if not all mission.scouts_deployed:
            if n_deployed != n_scouts:
                actions.append(_CMD_START_SCOUTS)
            else:
                actions.append(_CMD_RESTART_SCOUTS)
            # if any mission.scouts_deployed:
            if n_deployed:
                actions.append(_CMD_STOP_SCOUTS)
            # if all mission.scouts_deployed:
            if n_deployed == n_scouts:
                actions.append(_CMD_START_MISSION)
        else:
            mission_dir = Path(mission.mission_dir)
            if deployment.check_home(mission_dir):
                actions.append(_CMD_STOP_MISSION)
            else:
                actions.append(_CMD_RESET)

    if st.session_state.get("controls") is None:
        st.session_state.controls = None
    st.pills(
        "Mission control",
        actions,
        key="controls",
        default=None,
        on_change=_action,
        # on_change=_mission_controller,
        # args=(mission,),
    )


def mission_advanced_controls(mission: Mission) -> None:
    mission_active = mission.state() in ["Starting", "Running"]

    actions = []
    if not mission.is_template:
        actions.append(_CMD_CHECK_SCOUTS)
        actions.append(_CMD_START_SCOUTS)
        actions.append(_CMD_STOP_SCOUTS)
        actions.append(_CMD_STOP_MISSION)

        if not mission_active:
            actions.append(_CMD_DELETE)

    if not actions:
        return

    st.pills(
        "Advanced mission control",
        actions,
        key="advanced_controls",
        default=None,
        on_change=_action,
        # on_change=_mission_controller,
        # args=(mission,),
        label_visibility="collapsed",
    )
