# SPDX-FileCopyrightText: 2025-2026 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import shutil
import zipfile
from collections import defaultdict
from io import BytesIO
from typing import TYPE_CHECKING, Iterator

import streamlit as st
from PIL import Image

from hawk.gui.elements import columns, empty_state, load_mission, page_header

if TYPE_CHECKING:
    from pathlib import Path

    from streamlit.runtime.uploaded_file_manager import UploadedFile

EXTRACT_CMD = "extract"
REPACK_CMD = "repack"
DELETE_CMD = "discard"
IMAGE_SUFFIXES = (".gif", ".png", ".jpg", ".jpeg")
PREVIEW_IMAGES = 12

mission = load_mission()
mission_state = mission.state()
editable = mission_state == "Not Started" and not mission.is_template

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
extracted = bootstrap_dir.is_dir()


def _class_name(class_dir_name: str) -> str:
    try:
        return classes[int(class_dir_name)]
    except (IndexError, ValueError):
        return class_dir_name


###
# actions
def _action(action: str) -> None:
    st.session_state["bootstrap_action"] = action


action = st.session_state.pop("bootstrap_action", None)

if action == EXTRACT_CMD:
    try:
        with st.spinner("Extracting bootstrap examples…"):
            bootstrap_dir.mkdir()
            shutil.unpack_archive(bootstrap_zip, extract_dir=bootstrap_dir)
    except (OSError, ValueError):
        st.error("Failed to extract bootstrap.")
        shutil.rmtree(bootstrap_dir, ignore_errors=True)
    st.rerun()

if action == REPACK_CMD:
    with st.spinner("Saving bootstrap examples…"):
        bootstrap_zip.unlink(missing_ok=True)
        shutil.make_archive(
            str(bootstrap_zip.parent / bootstrap_zip.stem),
            format="zip",
            root_dir=bootstrap_dir,
        )
        shutil.rmtree(bootstrap_dir, ignore_errors=True)
    st.toast("Bootstrap examples saved", icon=":material/check_circle:")
    st.rerun()

if action == DELETE_CMD:
    with st.spinner("Discarding changes…"):
        shutil.rmtree(bootstrap_dir, ignore_errors=True)
    st.rerun()


actions = page_header("Bootstrap examples", mission)
with actions:
    if bootstrap_zip.exists() and not extracted:
        st.button(
            "Edit examples",
            icon=":material/edit:",
            on_click=_action,
            args=(EXTRACT_CMD,),
            help="Extract the bootstrap archive so examples can be added or "
            "removed" + ("" if editable else " (read-only for this mission)"),
        )
    if bootstrap_zip.exists() and extracted:
        st.button(
            "Discard changes",
            icon=":material/undo:",
            type="tertiary",
            on_click=_action,
            args=(DELETE_CMD,),
        )
    if extracted and editable:
        st.button(
            "Save changes",
            icon=":material/save:",
            type="primary",
            on_click=_action,
            args=(REPACK_CMD,),
            help=f"Repack the examples into {bootstrap_zip.name}",
        )


# delete image callback
def _delete(class_name: str, image_name: str) -> None:
    image_path = bootstrap_dir.joinpath(class_name, image_name).resolve()
    if bootstrap_dir in image_path.parents:
        image_path.unlink(missing_ok=True)
    else:
        st.error(f"Invalid image: {class_name}/{image_name}")


def _unpack(new_examples: list[UploadedFile] | None) -> Iterator[tuple[str, BytesIO]]:
    for example in new_examples or []:
        if not zipfile.is_zipfile(example):
            yield example.name, example
        else:
            with zipfile.ZipFile(example) as z:
                for name in z.namelist():
                    if not name.lower().endswith(IMAGE_SUFFIXES):
                        continue
                    with z.open(name) as item:
                        yield name.split("/")[-1], BytesIO(item.read())


def zip_examples() -> dict[str, list[str]]:
    """Map class directory names to images in the bootstrap archive."""
    examples: dict[str, list[str]] = defaultdict(list)
    with zipfile.ZipFile(bootstrap_zip) as z:
        for name in z.namelist():
            parts = name.split("/")
            if len(parts) >= 2 and name.lower().endswith(IMAGE_SUFFIXES):
                examples[parts[-2]].append(name)
    return dict(sorted(examples.items()))


def dir_examples() -> dict[str, list[Path]]:
    """Map class directory names to images in the extracted bootstrap."""
    return {
        class_dir.name: sorted(class_dir.iterdir())
        for class_dir in sorted(bootstrap_dir.iterdir())
        if class_dir.is_dir()
    }


def show_archive_preview() -> None:
    examples = zip_examples()
    if not examples:
        empty_state("The bootstrap archive is empty", icon=":material/folder_off:")
        return

    tabs = st.tabs(
        [f"{_class_name(name)} · {len(images)}" for name, images in examples.items()],
    )
    with zipfile.ZipFile(bootstrap_zip) as z:
        for tab, images in zip(tabs, examples.values()):
            with tab:
                column = columns(6)
                for name in images[:PREVIEW_IMAGES]:
                    with next(column):
                        st.image(BytesIO(z.read(name)), width="stretch")
                if len(images) > PREVIEW_IMAGES:
                    st.caption(
                        f"Showing {PREVIEW_IMAGES} of {len(images)} examples, "
                        "use **Edit examples** to see all of them.",
                    )


def show_extracted() -> int:
    """Show extracted examples, returns the number of positive examples."""
    examples = dir_examples()
    if not examples:
        empty_state(
            "No examples yet",
            "Upload example images for each class below.",
            icon=":material/add_photo_alternate:",
        )
        return 0

    delete = editable and st.toggle(
        "Remove examples",
        help="Show a delete button on every example",
    )
    tabs = st.tabs(
        [f"{_class_name(name)} · {len(images)}" for name, images in examples.items()],
    )
    for tab, (class_dir_name, images) in zip(tabs, examples.items()):
        with tab:
            column = columns(6)
            for image_file in images:
                with next(column), st.container(border=delete, gap="xsmall"):
                    st.image(str(image_file), width="stretch")
                    if delete:
                        st.button(
                            "",
                            icon=":material/delete:",
                            type="tertiary",
                            key=f"{class_dir_name}_{image_file.name}",
                            on_click=_delete,
                            args=(class_dir_name, image_file.name),
                            help=f"Remove {image_file.name}",
                        )
    return sum(
        len(images)
        for name, images in examples.items()
        if _class_name(name) != "negative"
    )


def upload_examples(positives: int) -> None:
    with st.expander(
        "Add examples",
        icon=":material/add_photo_alternate:",
        expanded=positives == 0,
    ):
        st.caption(
            "Images are cropped to a centered square and resized to 256×256. "
            "Zip archives are unpacked.",
        )
        with st.form("upload_examples", clear_on_submit=True, border=False):
            new_class = st.selectbox("Class", classes, index=1)
            new_examples = st.file_uploader(
                "Examples",
                type=["gif", "png", "jpg", "jpeg", "zip"],
                accept_multiple_files=True,
            )
            submitted = st.form_submit_button(
                "Upload",
                icon=":material/upload:",
                type="primary",
            )

        if submitted and new_examples:
            class_dir = bootstrap_dir / str(classes.index(new_class))
            class_dir.mkdir(exist_ok=True)
            count = 0
            for name, example in _unpack(new_examples):
                img = Image.open(example)
                # crop to centered square and resize to 256 by 256
                size = min(img.size)
                left = (img.size[0] - size) // 2
                top = (img.size[1] - size) // 2
                img = img.crop((left, top, left + size, top + size))
                img = img.resize((256, 256))
                img.save(class_dir / name)
                count += 1
            st.toast(
                f"Added {count} {new_class} example{'s' if count != 1 else ''}",
                icon=":material/check_circle:",
            )
            st.rerun()


if extracted:
    if editable:
        st.info(
            "You are editing the extracted examples. **Save changes** to repack "
            f"them into `{bootstrap_zip.name}` before starting the mission.",
            icon=":material/edit:",
        )
    positives = show_extracted()
    if editable:
        upload_examples(positives)
elif bootstrap_zip.exists():
    show_archive_preview()
elif editable:
    empty_state(
        "No bootstrap examples",
        "Upload a bootstrap archive on the Setup page, or start one here.",
        icon=":material/photo_library:",
    )
    if st.button("Create bootstrap examples", icon=":material/add:"):
        bootstrap_dir.mkdir()
        st.rerun()
else:
    empty_state(
        "This mission has no bootstrap examples", icon=":material/photo_library:"
    )
