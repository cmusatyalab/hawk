#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only
"""Build a Hawk search-and-rescue mission dataset from C2A and AIDER.

Hawk's dnn_classifier labels whole tiles, one class per tile, so this script
turns the two public datasets into tiles with exactly one label each:

* C2A (Nihal et al., 2024): AIDER disaster photos with synthetic people pasted
  in. Every person has a box and one of five poses. We cut each image into a
  grid of tiles. A tile that fully contains people of one pose gets that pose.
  A tile with no people is a negative.
* AIDER (Kyrkou and Theocharides, 2020): real aerial disaster photos. We tile
  the "normal" scenes (not used as C2A backgrounds) and use them as extra
  negatives, so that people become rare in the stream, like in a real search.

Outputs, under --out:

    tiles/...              the tile images that are used
    index/stream.txt       Hawk stream file: "<path> <label>", shuffled
                           (paths are relative to --out, which is the parent of
                           index/, as Hawk's random retriever expects)
    bootstrap.zip          folders 0/ (negative) and 1/..5/ (poses)
    mission_snippet.yml    the dataset/bootstrap part of a Hawk mission config
    stats.json             what was kept, dropped and why

Labels: 0 = negative, 1 = bent, 2 = kneeling, 3 = lying, 4 = sitting,
5 = upright, matching `dataset.class_list` in the snippet.

Only the standard library and Pillow are needed.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


POSES = ["bent", "kneeling", "lying", "sitting", "upright"]  # C2A pose ids 0..4
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
C2A_POSE_LABEL_DIRS = ["All labels with Pose information", "All labels with Pose info"]
C2A_SPLITS = ["train", "val", "test"]


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    pose: int


# ---------------------------------------------------------------- reading


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )


def read_c2a_boxes(label_file: Path, width: int, height: int) -> list[Box]:
    """Parse one 'All labels with Pose information' file.

    Each line is: class x_center y_center width height pose (normalized).
    """
    boxes = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 6:
            msg = f"{label_file}: expected 6 fields with pose, got {line!r}"
            raise ValueError(msg)
        xc, yc, w, h = (float(v) for v in parts[1:5])
        pose = int(float(parts[5]))
        if not 0 <= pose < len(POSES):
            msg = f"{label_file}: unknown pose id {pose}"
            raise ValueError(msg)
        boxes.append(
            Box(
                (xc - w / 2) * width,
                (yc - h / 2) * height,
                (xc + w / 2) * width,
                (yc + h / 2) * height,
                pose,
            ),
        )
    return boxes


def find_c2a_images(c2a_root: Path) -> list[tuple[str, Path, Path]]:
    """Return (split, image, pose label file) for every C2A image."""
    pose_dir = next(
        (c2a_root / d for d in C2A_POSE_LABEL_DIRS if (c2a_root / d).is_dir()),
        None,
    )
    if pose_dir is None:
        msg = f"no pose label folder ({C2A_POSE_LABEL_DIRS[0]!r}) in {c2a_root}"
        raise SystemExit(msg)
    pose_files = {p.stem: p for p in pose_dir.rglob("*.txt")}
    found = []
    missing = 0
    for split in C2A_SPLITS:
        image_dir = c2a_root / split / "images"
        if not image_dir.is_dir():
            continue
        for image in list_images(image_dir):
            label = pose_files.get(image.stem)
            if label is None:
                missing += 1
                continue
            found.append((split, image, label))
    if missing:
        print(f"warning: {missing} C2A images have no pose label file", file=sys.stderr)
    if not found:
        msg = f"no C2A images found under {c2a_root}/{{train,val,test}}/images"
        raise SystemExit(msg)
    return found


# ---------------------------------------------------------------- tiling


def grid(length: int, tile: int) -> list[int]:
    """Tile origins along one axis. The last tile is shifted back so it stays
    inside the image; an image smaller than a tile gets one (padded) tile."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def visible_fraction(box: Box, x: int, y: int, size: int) -> float:
    ix = max(0.0, min(box.x1, x + size) - max(box.x0, x))
    iy = max(0.0, min(box.y1, y + size) - max(box.y0, y))
    area = max(1e-9, (box.x1 - box.x0) * (box.y1 - box.y0))
    return ix * iy / area


def label_tile(
    boxes: list[Box],
    x: int,
    y: int,
    size: int,
    min_visible: float,
    mixed: str,
) -> tuple[int | None, str]:
    """Return (label, reason). label None means the tile is dropped."""
    full: Counter[int] = Counter()
    partial = 0
    for box in boxes:
        v = visible_fraction(box, x, y, size)
        if v >= min_visible:
            full[box.pose] += 1
        elif v > 0:
            partial += 1
    if not full:
        # only cut-off people: we can't say what the tile shows, so drop it
        return (None, "partial_only") if partial else (0, "empty")
    if len(full) == 1:
        return next(iter(full)) + 1, "single_pose"
    if mixed == "drop":
        return None, "mixed_pose"
    # majority pose, ties dropped
    (top, n1), (_, n2) = full.most_common(2)
    if n1 == n2:
        return None, "mixed_pose_tie"
    return top + 1, "mixed_majority"
