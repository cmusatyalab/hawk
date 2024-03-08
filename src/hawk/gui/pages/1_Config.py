# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import base64
import time

import pandas as pd
import streamlit as st

from hawk.gui import deployment
from hawk.gui.elements import (
    HOME_MISSION_DIR,
    SCOUT_MISSION_DIR,
    page_header,
    save_state,
)
from hawk.mission_config import MissionConfig

page_header("Configure Hawk Mission")
st.title("Configure Hawk Mission")


TRAIN_STRATEGY_DEFAULTS: dict[str, dict[str, str]] = {
    "dnn_classifier": {
        "mode": "hawk",
        "arch": "resnet50",
        "unfreeze_layers": "3",
        "online_epochs": "[[10,0],[15,100]]",
    },
    "fsl": {
        "mode": "hawk",
        "support_path": "/srv/diamond/dota/support.jpg",
        "fsl_traindir": "/srv/diamond/dota/fsl_traindir",
        "support_data": "",
    },
    "yolo": {
        "mode": "hawk",
        "online_epochs": "[[10,0],[15,100]]",
    },
    "custom": {},
}

RETRAIN_POLICY_DEFAULTS: dict[str, dict[str, bool | float | int | str]] = {
    "absolute": {"threshold": 0.33, "only_positives": True},
    "model": {},
    "percentage": {"threshold": 0.33, "only_positives": True},
    "sample": {"num_intervals": 10},
}

REEXAMINATION_DEFAULTS: dict[str, dict[str, bool | float | int | str]] = {
    "none": {},
    "top": {"k": 100},
    "full": {},
}

SELECTOR_DEFAULTS: dict[str, dict[str, float | int]] = {
    # "diversity": {
    #     "k": 10,
    #     "batchSize": 1000,
    #     "countermeasure_threshold": 0.5,
    #     "total_countermeasures": 0,
    # },
    # "threshold": {"threshold": 0.5},
    "topk": {
        "k": 10,
        "batchSize": 1000,
        "countermeasure_threshold": 0.5,
        "total_countermeasures": 0,
    },
    "token": {
        "initial_samples": 100,
        "batch_size": 1000,
        "countermeasure_threshold": 0.5,
        "total_countermeasures": 0,
    },
}


def change_train_strategy(
    train_type: str | None = None, train_args: dict[str, str] | None = None
) -> None:
    if train_type is None:
        train_type = st.session_state.train_strategy_type
        if train_type == "custom":
            train_type = st.session_state.train_strategy_custom_type
    assert train_type is not None
    train_strategy = st.session_state.mission_config.setdefault("train_strategy", {})
    train_strategy["type"] = train_type
    train_strategy["args"] = TRAIN_STRATEGY_DEFAULTS.get(
        train_type, {"mode": "hawk"}
    ).copy()
    if train_args is not None:
        train_strategy["args"].update(train_args)


def update_config(section: str | None, field: str) -> None:
    if section is not None:
        value = st.session_state[f"{section}_{field}"]
        if field == "type":
            if section == "retrain_policy":
                defaults = RETRAIN_POLICY_DEFAULTS
            elif section == "reexamination":
                defaults = REEXAMINATION_DEFAULTS
            else:
                defaults = {}
            st.session_state.mission_config[section] = defaults.get(value, {}).copy()
        st.session_state.mission_config[section][field] = value
    else:
        st.session_state.mission_config[field] = st.session_state[field]


def update_selector_config(selector_type: str | None, field: str) -> None:
    value = st.session_state[f"selector_{field}"]
    if field == "type":
        st.session_state.mission_config["selector"] = {
            "type": value,
            value: SELECTOR_DEFAULTS.get(value, {}),
        }
    else:
        st.session_state.mission_config["selector"][selector_type][field] = value


def set_state_from_mission_config(mission_config: MissionConfig) -> None:
    """Called whenever we reset/reload mission state to update st.session_state"""
    st.session_state.mission_config = mission_config

    st.session_state.scouts = [str(scout) for scout in mission_config.deploy.scouts]
    st.session_state.deployed = [False] * len(st.session_state.scouts)
    st.session_state._scout_port = mission_config.deploy.a2s_port

    # fix default values
    mission_config.setdefault("mission-name", "")
    mission_config["train-location"] = "scout"
    mission_config["label-mode"] = "ui"
    mission_config["home-params"] = {"mission_dir": str(HOME_MISSION_DIR)}
    mission_config["scout-params"] = {"mission_dir": str(SCOUT_MISSION_DIR)}
    mission_config.setdefault("dataset", {})

    train_strategy = mission_config.get("train_strategy", {})
    train_strategy_type = train_strategy.get("type", "dnn_classifier")
    change_train_strategy(train_strategy_type, train_strategy.get("args"))

    mission_config.setdefault(
        "retrain_policy",
        {"type": "percentage", "threshold": 0.33, "only_positives": True},
    )
    mission_config.setdefault("reexamination", {"type": "top", "k": 100})

    selector_type = mission_config.setdefault("selector", {"type": "topk"})["type"]
    selector_config = SELECTOR_DEFAULTS.get(selector_type, {}).copy()
    selector_config.update(mission_config["selector"].get(selector_type, {}))
    mission_config["selector"] = {
        "type": selector_type,
        selector_type: selector_config,
    }


def load_template() -> None:
    template_config = st.session_state.template_config
    if template_config is not None:
        mission_config = MissionConfig.from_yaml(
            template_config.getvalue().decode("utf-8")
        )
    else:
        mission_config = MissionConfig.from_dict({})
    set_state_from_mission_config(mission_config)


if "mission_config" not in st.session_state:
    mission_config = MissionConfig.from_dict({})
    set_state_from_mission_config(mission_config)

mission_name = st.session_state.mission_config["mission-name"]
with st.expander(f":floppy_disk: {mission_name}", expanded=not mission_name):
    st.session_state["mission-name"] = mission_name
    st.text_input(
        r"Mission name :red[\*]",
        key="mission-name",
        on_change=update_config,
        args=(
            None,
            "mission-name",
        ),
    )
    st.file_uploader(
        "Upload a template Hawk mission config",
        type=["yml"],
        key="template_config",
        on_change=load_template,
    )
    st.markdown("or configure a new mission below")

all_scouts_deployed = (
    st.session_state.get("scouts") and False not in st.session_state.deployed
)
status = ":ballot_box_with_check:" if all_scouts_deployed else ":o:"
with st.expander(f"{status} Scout Deployment", expanded=not all_scouts_deployed):
    ####
    # There has got to be an easier way to handle the st.data_editor...
    if "scouts" not in st.session_state:
        st.session_state.scouts = []
        st.session_state.deployed = []

    def update_scouts() -> None:
        for row in reversed(st.session_state.scout_edits["deleted_rows"]):
            del st.session_state.scouts[row]
            del st.session_state.deployed[row]
        for row in st.session_state.scout_edits["added_rows"]:
            if "host" in row:
                st.session_state.scouts.append(row["host"])
                st.session_state.deployed.append(False)
        for row, value in st.session_state.scout_edits["edited_rows"].items():
            st.session_state.scouts[int(row)] = value["host"]

    st.caption(r"Scouts :red[\*]")
    st.data_editor(
        pd.DataFrame(
            {
                "host": pd.Series(st.session_state.scouts, dtype="string"),
                "deployed": pd.Series(st.session_state.deployed, dtype=bool),
            }
        ),
        column_config={
            "host": st.column_config.TextColumn(
                "Host",
                help="SSH host used to deploy the scout",
            ),
            "deployed": st.column_config.CheckboxColumn(
                "Deployed",
                disabled=True,
                default=False,
            ),
        },
        num_rows="dynamic",
        hide_index=True,
        key="scout_edits",
        on_change=update_scouts,
    )

    st.session_state.scout_port = st.session_state.get("_scout_port", 6100)
    st.number_input(
        "Scout port",
        min_value=1024,
        max_value=65535,
        key="scout_port",
        on_change=save_state,
        args=("scout_port",),
    )

    col1, col2, col3 = st.columns(3)
    col1.button("Deploy scouts", on_click=deployment.start)
    col2.button("Stop scouts", on_click=deployment.stop)
    col3.button("Check deployment", on_click=deployment.check)

with st.expander("Dataset"):
    dataset_config = st.session_state.mission_config.get("dataset", {})
    st.session_state.dataset_type = dataset_config.get("type", "random")
    st.selectbox(
        "Dataset Type",
        ["cookie", "frame", "random", "tile", "video"],
        key="dataset_type",
        # on_change=change_dataset_type,
    )

    st.session_state.dataset_index_path = dataset_config.get("index_path")
    st.text_input(
        r"Index Path :red[\*]",
        key="dataset_index_path",
        on_change=update_config,
        args=("dataset", "index_path"),
    )

    st.session_state.dataset_tiles_per_frame = dataset_config.get(
        "tiles_per_frame", 100
    )
    st.number_input(
        "Tiles per frame",
        key="dataset_tiles_per_frame",
        min_value=1,
        on_change=update_config,
        args=("dataset", "tiles_per_frame"),
    )

    def save_dataset_stream() -> None:
        uploaded_file = st.session_state.dataset_stream
        st.session_state.mission_config["dataset"]["stream_path"] = (
            uploaded_file.name if uploaded_file is not None else None
        )

    st.file_uploader(
        rf"Stream file :red[\*]\: {dataset_config.get('stream_path', '')}",
        type=["txt"],
        key="dataset_stream",
        on_change=save_dataset_stream,
    )

with st.expander("Mission Parameters", expanded=True):
    ###
    # Train Strategy
    with st.container(border=True):
        train_config = st.session_state.mission_config["train_strategy"]
        train_type = train_config["type"]
        if train_type in ["dnn_classifier", "fsl", "yolo"]:
            st.session_state.train_strategy_type = train_type
            st.session_state.train_strategy_custom_type = ""
        else:
            st.session_state.train_strategy_type = "custom"
            st.session_state.train_strategy_custom_type = train_type

        col1, col2, col3 = st.columns(3)
        col1.selectbox(
            "Training Strategy",
            list(TRAIN_STRATEGY_DEFAULTS.keys()),
            key="train_strategy_type",
            on_change=change_train_strategy,
        )
        if st.session_state.train_strategy_type == "custom":
            col2.text_input(
                "Type",
                key="train_strategy_custom_type",
                on_change=change_train_strategy,
            )

        train_args = train_config["args"]

        def change_train_strategy_mode() -> None:
            st.session_state.mission_config["train_strategy"]["args"][
                "mode"
            ] = st.session_state.train_strategy_mode

        st.session_state.train_strategy_mode = train_args["mode"]
        col3.selectbox(
            "Mode",
            ["hawk", "oracle", "notional"],
            key="train_strategy_mode",
            on_change=change_train_strategy_mode,
        )

        def save_training_bootstrap() -> None:
            uploaded_file = st.session_state.training_bootstrap
            st.session_state.mission_config["train_strategy"]["bootstrap_path"] = (
                uploaded_file.name if uploaded_file is not None else None
            )

        st.file_uploader(
            r"Upload an archive with examples :red[\*]\:"
            f" {train_config.get('bootstrap_path', '')}",
            type=["zip"],
            key="training_bootstrap",
            on_change=save_training_bootstrap,
        )

        def save_training_initial_model() -> None:
            uploaded_file = st.session_state.training_initial_model
            st.session_state.mission_config["train_strategy"]["initial_model_path"] = (
                uploaded_file.name if uploaded_file is not None else None
            )

        st.file_uploader(
            rf"Upload an initial model\: {train_config.get('initial_model_path', '')}",
            type=["pth"],
            key="training_initial_model",
            on_change=save_training_initial_model,
        )

        st.caption("Training hyperparameters")
        st.session_state.train_strategy_args = {
            k: v for k, v in train_args.items() if k not in ["mode", "support_data"]
        }
        if train_args.get("support_data"):
            st.session_state.train_strategy_args["support_data"] = "..."

        st.data_editor(
            pd.Series(st.session_state.train_strategy_args, name="value"),
            num_rows="dynamic",
            hide_index=True,
            key="train_strategy_args_edits",
        )

        def change_train_strategy_support_data() -> None:
            support_data_file = st.session_state.train_strategy_support_data
            if support_data_file is not None:
                support_data_bytes = support_data_file.getvalue()
                support_data = base64.b64encode(support_data_bytes).decode("utf-8")
                train_config["args"]["support_data"] = support_data
            else:
                train_config["args"]["support_data"] = ""

        if train_type in ["fsl", "custom"]:
            st.file_uploader(
                "Upload support data",
                type=["jpg", "jpeg", "png"],
                key="train_strategy_support_data",
                on_change=change_train_strategy_support_data,
            )
    ###
    # Retrain Policy
    with st.container(border=True):
        retrain_policy_config = st.session_state.mission_config["retrain_policy"]
        st.session_state.retrain_policy_type = retrain_policy_config["type"]
        st.selectbox(
            "Retrain Policy Type",
            list(RETRAIN_POLICY_DEFAULTS.keys()),
            key="retrain_policy_type",
            on_change=update_config,
            args=("retrain_policy", "type"),
        )

        col1, col2, _, _ = st.columns(4)
        if "threshold" in retrain_policy_config:
            st.session_state.retrain_policy_threshold = retrain_policy_config[
                "threshold"
            ]
            col1.number_input(
                "Threshold",
                min_value=0.0,
                max_value=1.0,
                key="retrain_policy_threshold",
                on_change=update_config,
                args=("retrain_policy", "threshold"),
            )

        if "only_positives" in retrain_policy_config:
            st.session_state.retrain_policy_only_positives = retrain_policy_config[
                "only_positives"
            ]
            col2.caption("Only Positives")
            col2.toggle(
                "Only Positives",
                key="retrain_policy_only_positives",
                on_change=update_config,
                args=("retrain_policy", "only_positives"),
                label_visibility="hidden",
            )

        if "num_intervals" in retrain_policy_config:
            st.session_state.retrain_policy_num_intervals = retrain_policy_config[
                "num_intervals"
            ]
            col1.number_input(
                "Num Intervals",
                min_value=0,
                key="retrain_policy_num_intervals",
                on_change=update_config,
                args=("retrain_policy", "num_intervals"),
            )
    ###
    # Reexamination
    with st.container(border=True):
        reexamination_config = st.session_state.mission_config["reexamination"]
        st.session_state.reexamination_type = reexamination_config["type"]
        st.selectbox(
            "Reexamination Type",
            list(REEXAMINATION_DEFAULTS.keys()),
            key="reexamination_type",
            on_change=update_config,
            args=("reexamination", "type"),
        )

        col1, _, _, _ = st.columns(4)
        if "k" in reexamination_config:
            st.session_state.reexamination_k = reexamination_config["k"]
            col1.number_input(
                "K",
                min_value=1,
                key="reexamination_k",
                on_change=update_config,
                args=("reexamination", "k"),
            )
    ###
    # Selector
    with st.container(border=True):
        selector_type = st.session_state.mission_config["selector"]["type"]
        st.session_state.selector_type = selector_type
        st.selectbox(
            "Selector Type",
            list(SELECTOR_DEFAULTS.keys()),
            key="selector_type",
            on_change=update_selector_config,
            args=(None, "type"),
        )

        selector_config = st.session_state.mission_config["selector"].get(
            selector_type, {}
        )

        col1, col2, col3, col4 = st.columns(4)
        if "threshold" in selector_config:
            st.session_state.selector_threshold = selector_config["threshold"]
            col1.number_input(
                "Threshold",
                min_value=0.0,
                max_value=1.0,
                key="selector_threshold",
                on_change=update_selector_config,
                args=(selector_type, "threshold"),
            )

        if "initial_samples" in selector_config:
            st.session_state.selector_initial_samples = selector_config[
                "initial_samples"
            ]
            col1.number_input(
                "Initial Samples",
                min_value=0,
                key="selector_initial_samples",
                on_change=update_selector_config,
                args=(selector_type, "initial_samples"),
            )

        if "k" in selector_config:
            st.session_state.selector_k = selector_config["k"]
            col1.number_input(
                "K",
                min_value=1,
                key="selector_k",
                on_change=update_selector_config,
                args=(selector_type, "k"),
            )

        if "batchSize" in selector_config:  # diversity and topk use batchSize
            st.session_state.selector_batchSize = selector_config["batchSize"]
            col2.number_input(
                "Batch Size",
                min_value=1,
                key="selector_batchSize",
                on_change=update_selector_config,
                args=(selector_type, "batchSize"),
            )

        if "batch_size" in selector_config:  # token uses batch_size
            st.session_state.selector_batch_size = selector_config["batch_size"]
            col2.number_input(
                "Batch Size",
                min_value=1,
                key="selector_batch_size",
                on_change=update_selector_config,
                args=(selector_type, "batch_size"),
            )

        if "countermeasure_threshold" in selector_config:
            st.session_state.selector_countermeasure_threshold = selector_config[
                "countermeasure_threshold"
            ]
            col3.number_input(
                "Countermeasure Threshold",
                min_value=0.0,
                max_value=1.0,
                key="selector_countermeasure_threshold",
                on_change=update_selector_config,
                args=(selector_type, "countermeasure_threshold"),
            )

            st.session_state.selector_total_countermeasures = selector_config[
                "total_countermeasures"
            ]
            col4.number_input(
                "Total Countermeasures",
                min_value=0,
                key="selector_total_countermeasures",
                on_change=update_selector_config,
                args=(selector_type, "total_countermeasures"),
            )

with st.expander("Debug"):
    st.write(st.session_state)


def start_mission(mission_config: MissionConfig) -> None:
    now = int(time.time())
    mission_config["start-time"] = now

    # mission_start = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
    # result_dir = HOME_MISSION_DIR.joinpath(
    #     f"{mission_config['mission-name']}-{mission_start}"
    # )
    # result_dir.mkdir()
    # result_dir.write_text(mission_config.to_yaml())
