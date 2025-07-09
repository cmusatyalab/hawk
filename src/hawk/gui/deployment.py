# SPDX-FileCopyrightText: 2024-2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
from pathlib import Path

import streamlit as st

from hawk.deploy.build import build, export_requirements
from hawk.deploy.deploy import (
    check_deployment,
    deploy,
    restart_deployment,
    stop_deployment,
)
from hawk.deploy_config import DeployConfig, SshHost


def prune_dead_scouts(deploy_config: DeployConfig) -> None:
    if not st.session_state.get("deployed_state"):
        st.session_state.deployed_state = []
        return

    # ignore those that we will actively redeploy/restart/check
    for scout in deploy_config.scouts:
        with contextlib.suppress(ValueError):
            st.session_state.deployed_state.remove(str(scout))

    # stop whatever is left...
    if st.session_state.deployed_state:
        scouts = [SshHost.from_str(scout) for scout in st.session_state.deployed_state]
        deploy_config = DeployConfig(scouts=scouts)
        stop_scouts(deploy_config)


def deploy_scouts(deploy_config: DeployConfig) -> bool:
    prune_dead_scouts(deploy_config)

    with st.status("Deploying scouts", expanded=True) as status:
        st.text("Building Hawk wheel...")
        dist_wheel = build()
        if not dist_wheel.exists():
            status.update(label="Building Hawk wheel failed", state="error")
            st.stop()

        st.text("Exporting requirements.txt...")
        dist_requirements = export_requirements()
        if not dist_requirements.exists():
            status.update(label="Exporting requirement.txt failed", state="error")
            st.stop()

        st.text("Deploying to scouts...")
        if deploy(deploy_config, dist_wheel, dist_requirements) == 0:
            status.update(label="Deployed scouts", state="complete", expanded=False)
            # shortcut, we shouldn't need to check if we successfully deployed
            st.session_state.deployed_state = [
                str(scout) for scout in deploy_config.scouts
            ]
        else:
            status.update(label="Deployment failed", state="error")

    return check_scouts(deploy_config, quick=True)


def restart_scouts(deploy_config: DeployConfig) -> bool:
    prune_dead_scouts(deploy_config)

    with st.status("Restarting scouts", expanded=True) as status:
        st.text("Restarting scouts...")
        if restart_deployment(deploy_config) == 0:
            status.update(label="Scouts restarted", state="complete", expanded=False)
            # shortcut, we shouldn't need to check if we successfully restarted
            st.session_state.deployed_state = [
                str(scout) for scout in deploy_config.scouts
            ]
        else:
            status.update(label="Restarting scouts failed", state="error")

    return check_scouts(deploy_config)


def stop_scouts(deploy_config: DeployConfig) -> bool:
    with st.status("Stopping scouts", expanded=True) as status:
        st.text("Stopping scouts...")
        stop_deployment(deploy_config)
        status.update(label="Scouts stopped", state="complete", expanded=False)

    if st.session_state.deployed_state:
        prune_dead_scouts(deploy_config)
        st.session_state.deployed_state = []
    return True


def check_scouts(deploy_config: DeployConfig, quick: bool = False) -> bool:
    deployed = st.session_state.get("deployed_state", [])
    if quick and len(deployed) == len(deploy_config.scouts):
        return True

    prune_dead_scouts(deploy_config)
    with st.status("Checking scouts", expanded=True) as status:
        for scout in deploy_config.scouts:
            st.text(f"Checking {scout}...")
            one_scout_config = DeployConfig(
                scouts=[scout],
                scout_port=deploy_config.scout_port,
            )
            rc = check_deployment(one_scout_config)
            if rc == 0:
                st.session_state.deployed_state.append(str(scout))
            else:
                st.text("...not deployed")
        ok = len(st.session_state.deployed_state) == len(deploy_config.scouts)
        if not ok:
            status.update(label="Not all scouts deployed", state="error")
        else:
            status.update(label="All scouts deployed", state="complete", expanded=False)
    return ok


def _get_home_pid(mission_dir: Path) -> int:
    pidfile = mission_dir.joinpath("hawk_home.pid")
    if not pidfile.exists():
        raise ValueError

    pid = int(pidfile.read_text())
    try:
        os.kill(pid, 0)
    except OSError as err:
        raise ValueError from err
    return pid


def check_home(mission_dir: Path) -> bool:
    try:
        _get_home_pid(mission_dir)
        return True
    except ValueError:
        return False


def start_home(mission_dir: Path) -> None:
    if check_home(mission_dir):
        stop_home(mission_dir)
    subprocess.run(
        [sys.executable, "-m", "hawk.home.home_main", "-d", mission_dir], check=True
    )


def stop_home(mission_dir: Path) -> None:
    with contextlib.suppress(ValueError):
        pid = _get_home_pid(mission_dir)
        os.kill(pid, signal.SIGTERM)
