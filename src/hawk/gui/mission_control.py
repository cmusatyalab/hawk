# SPDX-FileCopyrightText: 2025-2026 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import contextlib
import shutil
import time
import zipfile
from typing import TYPE_CHECKING, Callable

import streamlit as st

from hawk.gui import deployment

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

    from hawk.deploy_config import DeployConfig
    from hawk.gui.elements import Mission

# Mission control commands that show up in multiple places
_CMD_CLONE = "Clone mission"
_CMD_START_MISSION = "Start mission"
_CMD_STOP_MISSION = "Stop mission"
_CMD_RESET = "Reset mission"
_CMD_DELETE = "Delete mission"
_CMD_CHECK_SCOUTS = "Check scouts"
_CMD_DEPLOY = "Deploy scouts"
_CMD_START_SCOUTS = "Start scouts"
_CMD_RESTART_SCOUTS = "Restart scouts"
_CMD_STOP_SCOUTS = "Stop scouts"

if st.session_state.get("deployed_state") is None:
    st.session_state.deployed_state = []


@st.dialog("Are you sure?", icon=":material/warning:")
def _confirm(
    callback: Callable[[Mission], bool],
    mission: Mission,
    prompt: str,
    label: str,
    warning: str | None = None,
) -> None:
    """Ask for confirmation, destructive actions (with a warning) require the
    user to type the mission name.
    """
    if warning is not None:
        st.warning(
            f"This will **{prompt}** for `{mission.name}`. {warning}",
            icon=":material/delete_forever:",
        )
        confirmation = st.text_input(
            f"Type **{mission.name}** to confirm",
            placeholder=mission.name,
        )
        allowed = confirmation == mission.name
    else:
        st.markdown(f"This will **{prompt}**.")
        allowed = True

    with st.container(horizontal=True, horizontal_alignment="right"):
        cancelled = st.button("Cancel", type="tertiary")
        confirmed = st.button(label, type="primary", disabled=not allowed)

    if cancelled:
        st.rerun()
    if confirmed and callback(mission):
        time.sleep(1)
        st.rerun()


@st.dialog("Working on it", icon=":material/settings:")
def _progress(callback: Callable[[DeployConfig], bool], mission: Mission) -> None:
    if callback(mission.config.deploy):
        time.sleep(1)
        st.rerun()


def archive_mission_state(mission: Mission) -> None:
    """Archive mission state."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    archive_path = mission.mission_dir.joinpath(timestamp).with_suffix(".zip")

    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive_paths = [
            "mission_config.yml",
            *mission.extra_config_files,
            "bootstrap",
            "hawk_home.log",
            "logs",
            "traces",
            "unlabeled.jsonl",
            "labeled.jsonl",
            "images",
            "novel.jsonl",
            "novel",
            "feature_vectors",
        ]
        for file in archive_paths:
            path = mission.mission_dir / file
            if path.is_dir():
                # archive.mkdir(path)
                for sub_path in path.iterdir():
                    if sub_path.is_file():
                        arcname = sub_path.relative_to(mission.mission_dir)
                        archive.write(sub_path, str(arcname))
            elif path.is_file():
                archive.write(path, file)


def clone_mission(mission: Mission) -> bool:
    """Create a new mission from an existing one."""
    mission_name = mission.config.get("mission-name", mission.name.lstrip("_"))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    new_mission_name = f"{mission_name}-{timestamp}"
    new_mission_dir = mission.mission_dir.parent.joinpath(new_mission_name)
    with st.status(f"Creating Mission {new_mission_name}...", expanded=True) as status:
        st.write("Creating mission directory...")
        time.sleep(1)
        new_mission_dir.mkdir()
        st.write("Copying mission configuration...")

        # copy config files
        for file in ["mission_config.yml", *mission.extra_config_files]:
            path = mission.mission_dir / file

            # cleanup absolute paths that were written to mission/logs/hawk.yml
            with contextlib.suppress(ValueError):
                file = str(path.relative_to(mission.mission_dir))
            new_path = new_mission_dir / file

            if path.exists() and path != new_path:
                st.write(f"Copying {file}...")
                shutil.copy(path, new_path)
                time.sleep(1)

        status.update(
            label=f"Mission {new_mission_name} created",
            state="complete",
            expanded=True,
        )
    st.session_state["mission_name"] = new_mission_name
    return True


def reset_mission(mission: Mission) -> bool:
    """Archive and reset mission state (labels/images/logs)."""
    with st.status("Resetting Mission state...", expanded=True) as status:
        st.write("Archiving mission state...")
        archive_mission_state(mission)

        st.write("Removing labeled/unlabeled data...")
        time.sleep(0.5)
        mission.mission_dir.joinpath("unlabeled.jsonl").unlink(missing_ok=True)
        mission.mission_dir.joinpath("labeled.jsonl").unlink(missing_ok=True)
        shutil.rmtree(mission.mission_dir / "images", ignore_errors=True)
        st.write("Removing novel class examples...")
        time.sleep(0.5)
        mission.mission_dir.joinpath("novel.jsonl").unlink(missing_ok=True)
        shutil.rmtree(mission.mission_dir / "novel", ignore_errors=True)
        shutil.rmtree(mission.mission_dir / "feature_vectors", ignore_errors=True)
        st.write("Removing logs...")
        time.sleep(0.5)
        mission.mission_dir.joinpath("hawk_home.log").unlink(missing_ok=True)
        shutil.rmtree(mission.mission_dir / "traces", ignore_errors=True)
        shutil.rmtree(mission.mission_dir / "logs", ignore_errors=True)
        status.update(label="Mission reset", state="complete", expanded=True)
    return True


def delete_mission(mission: Mission) -> bool:
    """Completely destroy all state and configuration."""
    with st.status("Deleting Mission...", expanded=True) as status:
        st.write("Removing labeled/unlabeled data...")
        time.sleep(0.2)
        st.write("Removing novel class examples...")
        time.sleep(0.2)
        st.write("Removing logs...")
        time.sleep(0.2)
        st.write("Removing archived state...")
        time.sleep(0.2)
        st.write("Removing bootstrap examples...")
        time.sleep(0.2)
        st.write("Removing mission config...")
        time.sleep(0.2)
        st.write("Removing mission directory...")
        time.sleep(0.2)
        shutil.rmtree(mission.mission_dir, ignore_errors=True)
        status.update(label="Mission deleted", state="complete", expanded=True)
    st.session_state["mission_name"] = None
    return True


@st.dialog("Starting mission", icon=":material/play_arrow:")
def start_home(mission: Mission) -> None:
    """Start the mission, if all scouts are deployed."""
    n_deployed = len(st.session_state.get("deployed_state", []))
    n_scouts = len(mission.config.deploy.scouts)
    if n_deployed != n_scouts:
        st.error("Not all scouts are deployed")

    with st.status("Starting Mission...", expanded=True) as status:
        st.write("Starting Hawk process...")
        deployment.start_home(mission.mission_dir)
        time.sleep(1)
        status.update(label="Mission started", state="complete", expanded=True)
    time.sleep(2)
    st.rerun()


@st.dialog("Stopping mission", icon=":material/stop:")
def stop_home(mission: Mission) -> None:
    """Stop the mission and scouts."""
    deployment.stop_scouts(mission.config.deploy)

    with st.status("Stopping Mission...", expanded=True) as status:
        st.write("Stopping Hawk process...")
        deployment.stop_home(mission.mission_dir)
        time.sleep(1)
        status.update(label="Mission terminated", state="complete", expanded=True)
    time.sleep(2)
    st.rerun()


def _run_command(mission: Mission, command: str) -> None:
    if command == _CMD_CLONE:
        _confirm(
            clone_mission,
            mission,
            "create a new mission with the same configuration",
            "Clone mission",
        )
    elif command == _CMD_CHECK_SCOUTS:
        _progress(deployment.check_scouts, mission)
    elif command == _CMD_DEPLOY:
        _progress(deployment.deploy_scouts, mission)
    elif command in (_CMD_START_SCOUTS, _CMD_RESTART_SCOUTS):
        _progress(deployment.restart_scouts, mission)
    elif command == _CMD_STOP_SCOUTS:
        _progress(deployment.stop_scouts, mission)
    elif command == _CMD_START_MISSION:
        start_home(mission)
    elif command == _CMD_STOP_MISSION:
        stop_home(mission)
    elif command == _CMD_RESET:
        _confirm(
            reset_mission,
            mission,
            "remove all labels, results and logs",
            "Reset mission",
            warning="The current state is first saved to a zip archive in the "
            "mission directory, the configuration is kept.",
        )
    elif command == _CMD_DELETE:
        _confirm(
            delete_mission,
            mission,
            "permanently delete the mission",
            "Delete mission",
            warning="All results, labels, logs, archives and the configuration "
            "are removed. This cannot be undone.",
        )


_ICONS = {
    _CMD_CLONE: ":material/content_copy:",
    _CMD_CHECK_SCOUTS: ":material/network_check:",
    _CMD_DEPLOY: ":material/rocket_launch:",
    _CMD_START_SCOUTS: ":material/power_settings_new:",
    _CMD_RESTART_SCOUTS: ":material/restart_alt:",
    _CMD_STOP_SCOUTS: ":material/power_off:",
    _CMD_START_MISSION: ":material/play_arrow:",
    _CMD_STOP_MISSION: ":material/stop:",
    _CMD_RESET: ":material/history:",
    _CMD_DELETE: ":material/delete_forever:",
}


def mission_controls(mission: Mission, container: DeltaGenerator) -> None:
    """Mission control buttons, the common actions are shown as buttons and
    everything else is available from an overflow menu.
    """
    mission_state = mission.state()
    home_running = deployment.check_home(mission.mission_dir)

    actions: list[str] = []
    if not mission.is_template:
        n_deployed = len(st.session_state.get("deployed_state", []))
        n_scouts = len(mission.config.deploy.scouts)
        if mission_state == "Not Started":
            actions.append(_CMD_DEPLOY)
            if n_deployed != n_scouts:
                actions.append(_CMD_START_SCOUTS)
            else:
                actions.append(_CMD_RESTART_SCOUTS)
            if n_deployed:
                actions.append(_CMD_STOP_SCOUTS)
            if n_deployed == n_scouts:
                actions.append(_CMD_START_MISSION)
        elif home_running:
            actions.append(_CMD_STOP_MISSION)

    more = [_CMD_CLONE]
    if not mission.is_template:
        more += [_CMD_CHECK_SCOUTS, _CMD_START_SCOUTS, _CMD_STOP_SCOUTS]
        if home_running:
            more.append(_CMD_STOP_MISSION)
        # when the mission has finished and Hawk home is not running we can
        # reset and/or delete the mission state
        if not mission.is_active and not home_running:
            more += [_CMD_RESET, _CMD_DELETE]

    command = None
    with container:
        for action in actions:
            primary = action in (_CMD_START_MISSION, _CMD_STOP_MISSION)
            if st.button(
                action,
                icon=_ICONS[action],
                type="primary" if primary else "secondary",
                key=f"cmd_{action}",
            ):
                command = action

        selected = st.menu_button(
            "",
            [cmd for cmd in more if cmd not in actions],
            icon=":material/more_horiz:",
            help="More mission actions",
            format_func=lambda cmd: f"{_ICONS[cmd]} {cmd}",
            key="more_actions",
        )
        command = command or selected

    if command is not None:
        _run_command(mission, command)

    # If we are not a template, and scouts were configured but we don't know
    # the current deployment state, force a recheck.
    if (
        not mission.is_template
        and mission.config.deploy.scouts
        and "deployed_state" not in st.session_state
    ):
        _progress(lambda m: deployment.check_scouts(m) or True, mission)
