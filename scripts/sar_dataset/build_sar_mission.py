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

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

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


@dataclass
class Tile:
    source: Path  # source image
    origin: str  # "c2a" or "aider"
    split: str  # c2a split, or "aider"
    x: int
    y: int
    size: int
    label: int  # 0 negative, 1..5 pose
    name: str = ""
    extra: dict = field(default_factory=dict)


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


def find_aider_images(aider_root: Path, classes: list[str]) -> list[Path]:
    wanted = {c.lower() for c in classes}
    images = [
        p
        for p in list_images(aider_root)
        if any(part.lower() in wanted for part in p.relative_to(aider_root).parts[:-1])
    ]
    if not images:
        msg = f"no AIDER images in folders named {sorted(wanted)} under {aider_root}"
        raise SystemExit(msg)
    return images


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


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--c2a-root", type=Path, required=True,
                    help="C2A folder that holds train/ val/ test/ and 'All labels with Pose information/'")
    ap.add_argument("--aider-root", type=Path, required=True,
                    help="extracted AIDER folder (one subfolder per class)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--aider-classes", default="normal",
                    help="comma list of AIDER class folders to use as negatives (default: normal)")
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--min-visible", type=float, default=0.5,
                    help="fraction of a person's box that must be inside a tile to count")
    ap.add_argument("--mixed", choices=["drop", "majority"], default="drop",
                    help="what to do with a tile that holds people of different poses")
    ap.add_argument("--positive-rate", type=float, default=0.05,
                    help="target share of tiles with a person in the stream")
    ap.add_argument("--c2a-empty-negatives", action=argparse.BooleanOptionalAction, default=True,
                    help="also use empty C2A tiles as negatives (default: yes)")
    ap.add_argument("--limit-images", type=int, default=0,
                    help="use at most this many C2A images (0 = all); for a quick subset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if not 0 < args.positive_rate < 1:
        ap.error("--positive-rate must be between 0 and 1")
    rng = random.Random(args.seed)
    stats: dict = {"args": {k: str(v) for k, v in vars(args).items()}}
    reasons: Counter[str] = Counter()

    # 1. label C2A tiles from the boxes (no pixels needed yet)
    c2a = find_c2a_images(args.c2a_root)
    rng.shuffle(c2a)
    if args.limit_images:
        c2a = c2a[: args.limit_images]
    c2a_tiles: list[Tile] = []
    people_per_image = []
    for split, image_path, label_path in c2a:
        with Image.open(image_path) as im:
            width, height = im.size
        boxes = read_c2a_boxes(label_path, width, height)
        people_per_image.append(len(boxes))
        for y in grid(height, args.tile_size):
            for x in grid(width, args.tile_size):
                label, why = label_tile(boxes, x, y, args.tile_size,
                                        args.min_visible, args.mixed)
                reasons[why] += 1
                if label is not None:
                    c2a_tiles.append(Tile(image_path, "c2a", split, x, y,
                                          args.tile_size, label))

    # 2. AIDER tiles are all negatives
    aider_images = find_aider_images(args.aider_root,
                                     [c.strip() for c in args.aider_classes.split(",")])
    aider_tiles: list[Tile] = []
    for image_path in aider_images:
        with Image.open(image_path) as im:
            width, height = im.size
        for y in grid(height, args.tile_size):
            for x in grid(width, args.tile_size):
                aider_tiles.append(Tile(image_path, "aider", "aider", x, y,
                                        args.tile_size, 0))

    held_out: set[Path] = set()  # bootstrap images, added later

    # 3. stream: all remaining positives, plus negatives up to the target rate
    positives = [t for t in c2a_tiles if t.label > 0 and t.source not in held_out]
    negatives = [t for t in aider_tiles if t.source not in held_out]
    if args.c2a_empty_negatives:
        negatives += [t for t in c2a_tiles if t.label == 0 and t.source not in held_out]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    want_neg = round(len(positives) * (1 - args.positive_rate) / args.positive_rate)
    if want_neg > len(negatives):
        # not enough negatives: keep all of them and drop positives instead
        keep_pos = round(len(negatives) * args.positive_rate / (1 - args.positive_rate))
        print(f"note: only {len(negatives)} negative tiles; keeping {keep_pos} of "
              f"{len(positives)} positive tiles to reach the target rate",
              file=sys.stderr)
        positives = positives[:keep_pos]
        want_neg = len(negatives)
    stream = positives + negatives[:want_neg]
    rng.shuffle(stream)

    # 4. stats
    def per_class(tiles: list[Tile]) -> dict[str, int]:
        c = Counter(t.label for t in tiles)
        return {"negative": c[0], **{p: c[i + 1] for i, p in enumerate(POSES)}}

    stats.update({
        "c2a_images": len(c2a),
        "c2a_people_per_image_mean": round(sum(people_per_image) / max(1, len(people_per_image)), 1),
        "aider_images": len(aider_images),
        "c2a_tile_outcomes": dict(reasons),
        "c2a_tiles_by_class": per_class(c2a_tiles),
        "aider_tiles": len(aider_tiles),
        "stream_by_class": per_class(stream),
        "stream_tiles": len(stream),
        "stream_positive_rate": round(sum(t.label > 0 for t in stream) / max(1, len(stream)), 4),
    })
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "args"}, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
