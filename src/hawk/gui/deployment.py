# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import streamlit as st

from hawk.deploy.build import poetry_build, poetry_export_requirements
from hawk.deploy.deploy import check_deployment, deploy, stop_deployment
from hawk.deploy_config import DeployConfig


def start() -> None:
    with st.status("Deploying scouts", expanded=True) as status:
        st.text("Building Hawk wheel...")
        dist_wheel = poetry_build()
        if not dist_wheel.exists():
            status.update(label="Building Hawk wheel failed", state="error")
            st.stop()

        st.text("Exporting requirements.txt...")
        dist_requirements = poetry_export_requirements()
        if not dist_requirements.exists():
            status.update(label="Exporting requirement.txt failed", state="error")
            st.stop()

        for index, scout in enumerate(st.session_state.scouts):
            st.text(f"Deploying to {scout}...")
            deploy_config = DeployConfig(
                scouts=[scout],
                scout_port=st.session_state.scout_port,
            )
            rc = deploy(deploy_config, dist_wheel, dist_requirements)
            st.session_state.deployed[index] = rc == 0
            if rc != 0:
                st.text("...failed deployment")
        if False in st.session_state.deployed:
            status.update(label="Deployment failed", state="error")
        else:
            status.update(label="Deployed scouts", state="complete", expanded=False)


def stop() -> None:
    with st.status("Stopping scouts", expanded=True) as status:
        st.text("Stopping scouts...")
        deploy_config = DeployConfig(
            scouts=st.session_state.scouts,
            scout_port=st.session_state.scout_port,
        )
        stop_deployment(deploy_config)
        st.session_state.deployed = [False] * len(st.session_state.scouts)
        status.update(label="Scouts stopped", state="complete", expanded=False)


def check() -> None:
    with st.status("Checking scouts", expanded=True) as status:
        for index, scout in enumerate(st.session_state.scouts):
            st.text(f"Checking {scout}...")
            deploy_config = DeployConfig(
                scouts=[scout],
                scout_port=st.session_state.scout_port,
            )
            rc = check_deployment(deploy_config)
            st.session_state.deployed[index] = rc == 0
            if rc != 0:
                st.text("...not deployed")
        if False in st.session_state.deployed:
            status.update(label="Not all scouts deployed", state="error")
        else:
            status.update(label="All scouts deployed", state="complete", expanded=False)
