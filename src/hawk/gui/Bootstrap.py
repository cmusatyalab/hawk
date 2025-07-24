# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import streamlit as st
from PIL import Image

from hawk.gui.elements import columns, load_mission

if TYPE_CHECKING:
    from pathlib import Path

st.title("Hawk Mission Bootstrap")

EXTRACT_CMD = "Extract bootstrap"
REPACK_CMD = "Repack bootstrap"
DELETE_CMD = "Discard changes"

mission = load_mission()
mission_state = mission.state()

# get list of class names
classes: list[str] = mission.config.get("dataset", {}).get("class_list", ["positive"])
if classes[0] != "negative":
    classes.insert(0, "negative")

# extract various paths
bootstrap_dir = mission.mission_dir / "bootstrap"
bootstrap_zip = mission.mission_dir / mission.config.get("train_strategy", {}).get(
    "bootstrap_path",
    "bootstrap.zip",
)

# control buttons
actions = []
if bootstrap_zip.exists() and not bootstrap_dir.is_dir():
    actions.append(EXTRACT_CMD)
if (
    bootstrap_dir.is_dir()
    and mission_state == "Not Started"
    and not mission.is_template
):
    actions.append(REPACK_CMD)
if bootstrap_zip.exists() and bootstrap_dir.is_dir():
    actions.append(DELETE_CMD)

if "bootstrap_action" not in st.session_state:
    st.session_state["bootstrap_action"] = None


# action callback
def _action() -> None:
    action = st.session_state["bootstrap_actions"]
    st.session_state["bootstrap_actions"] = None
    st.session_state["bootstrap_action"] = action


st.pills("Bootstrap actions", actions, key="bootstrap_actions", on_change=_action)

# process commands
action = st.session_state["bootstrap_action"] or []
st.session_state["bootstrap_action"] = []

if EXTRACT_CMD in action:
    try:
        with st.spinner("Extracting bootstrap..."):
            bootstrap_dir.mkdir()
            shutil.unpack_archive(bootstrap_zip, extract_dir=bootstrap_dir)
    except (OSError, ValueError):
        st.error("Failed to extract bootstrap.")
        shutil.rmtree(bootstrap_dir, ignore_errors=True)
    st.rerun()

if REPACK_CMD in action:
    with st.spinner("Repacking bootstrap..."):
        bootstrap_zip.unlink()
        shutil.make_archive(
            bootstrap_zip.parent / bootstrap_zip.stem,
            format="zip",
            root_dir=bootstrap_dir,
        )
        shutil.rmtree(bootstrap_dir, ignore_errors=True)
    st.rerun()

if DELETE_CMD in action:
    with st.spinner("Deleting extracted bootstrap..."):
        shutil.rmtree(bootstrap_dir, ignore_errors=True)
    st.rerun()


# delete image callback
def _delete(class_name: str, image_name: str) -> None:
    image_path = bootstrap_dir.joinpath(class_name, image_name).resolve()
    if bootstrap_dir in image_path.parents:
        image_path.unlink(missing_ok=True)
    else:
        st.error(f"Invalid image: {class_name}/{image_name}")


def _class_name(class_dir: Path) -> str:
    try:
        class_name = classes[int(class_dir.name)]
    except (IndexError, ValueError):
        class_name = class_dir.name
    return class_name


# display extracted bootstrap examples
if bootstrap_dir.is_dir():
    column = columns(4)
    delete = False

    if mission_state == "Not Started" and not mission.is_template:
        delete = st.toggle("Delete bootstrap examples")

    class_dirs = sorted(cls for cls in bootstrap_dir.iterdir() if cls.is_dir())
    positives = 0
    for class_dir in class_dirs:
        imgs = list(class_dir.iterdir())

        if class_dir != class_dirs[0]:
            positives += len(imgs)

        with st.expander(f"Bootstrap class {_class_name(class_dir)} ({len(imgs)})"):
            if delete:
                for image_file in imgs:
                    with next(column), st.container(border=True):
                        st.image(str(image_file))
                        st.button(
                            "",
                            icon=":material/delete:",
                            key=f"{class_dir.name}_{image_file.name}",
                            on_click=_delete,
                            args=(class_dir.name, image_file.name),
                        )
            else:
                st.image([str(img) for img in imgs])

    if mission_state == "Not Started" and not mission.is_template:
        with st.expander("Add new bootstrap examples", expanded=positives == 0):
            new_class = st.selectbox("Class", classes, index=1)
            new_examples = st.file_uploader(
                "Examples",
                type=["gif", "png", "jpg"],
                accept_multiple_files=True,
            )
            if st.button("Upload"):
                class_dir = bootstrap_dir / str(classes.index(new_class))
                class_dir.mkdir(exist_ok=True)
                for example in new_examples or []:
                    img = Image.open(example)
                    # crop to centered square and resize to 256 by 256
                    size = min(img.size)
                    left = (img.size[0] - size) // 2
                    top = (img.size[1] - size) // 2
                    img = img.crop((left, top, left + size, top + size))
                    img = img.resize((256, 256))
                    img.save(class_dir / example.name)
                st.rerun()
