#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only
"""Build a Hawk search-and-rescue mission dataset from C2A and AIDER.

Reproduce our mission_v1 data in one command (about 5 GB download, 6 GB of
extracted images, 1 GB of tiles):

    python build_sar_mission.py all --dest ~/sar_data

or step by step: `download`, `extract`, `build`, `validate` (see --help of
each).

Hawk's dnn_classifier labels whole tiles, one class per tile, so this script
turns the two public datasets into tiles with exactly one label each:

* C2A (Nihal et al., 2024): AIDER disaster photos with synthetic people pasted
  in. Every person has a box and one of five poses. We cut each image into a
  grid of tiles. A tile that shows people of one pose (at least half of each
  box inside the part of the tile the classifier sees after its center crop)
  gets that pose. A tile with no people is a negative.
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
import hashlib
import json
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

POSES = ["bent", "kneeling", "lying", "sitting", "upright"]  # C2A pose ids 0..4
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
C2A_POSE_LABEL_DIRS = ["All labels with Pose information", "All labels with Pose info"]
C2A_SPLITS = ["train", "val", "test"]

# Where the two datasets come from. MD5 sums are of the files we used.
AIDER_URL = "https://zenodo.org/records/3888300/files/AIDER.zip?download=1"
AIDER_ZIP = "AIDER.zip"
AIDER_MD5 = "1ad4eb02ed156e8dfa19986ff382e58b"  # 275,701,096 bytes
# The Google Drive file linked from github.com/Ragib-Amin-Nihal/C2A
C2A_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1Uba6CHJRCvF-rgfCXDR6NhTnOvngWsfe&export=download&confirm=t"
)
C2A_ZIP = "C2A_Dataset.zip"
C2A_MD5 = "d836a206fb7c682c893fd47665ea24f1"  # 4,901,971,513 bytes
C2A_MANUAL = (
    "Download it by hand from https://github.com/Ragib-Amin-Nihal/C2A (Google Drive)\n"
    "or https://www.kaggle.com/datasets/rgbnihal/c2a-dataset, save it as\n"
    "{path}, and run this step again (add --no-verify for the Kaggle zip)."
)

# Only these parts of the archives are needed.
AIDER_MEMBERS = ["AIDER/normal/"]
C2A_MEMBERS = [
    "new_dataset3/All labels with Pose information/",
    "new_dataset3/train/images/",
    "new_dataset3/val/images/",
    "new_dataset3/test/images/",
]

# The settings of our mission_v1 build; `build` uses them by default.
DEFAULT_TILE_SIZE = 128
DEFAULT_POSE_MIX = "upright=0.4,sitting=0.25,bent=0.15,kneeling=0.15,lying=0.05"


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
    name: str = ""  # path of the written tile, relative to --out


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


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
    margin: int = 0,
) -> tuple[int | None, str]:
    """Return (label, reason). label None means the tile is dropped.

    A person counts only by the part of their box inside the tile minus
    `margin` pixels on each edge, the border the classifier crops away."""
    visible: Counter[int] = Counter()  # people per pose that count for the tile
    cut_off = 0
    inner = size - 2 * margin
    for box in boxes:
        if visible_fraction(box, x + margin, y + margin, inner) >= min_visible:
            visible[box.pose] += 1
        elif visible_fraction(box, x, y, size) > 0:
            cut_off += 1
    if not visible:
        # only cut-off people: we can't say what the tile shows, so drop it
        return (None, "partial_only") if cut_off else (0, "empty")
    if len(visible) == 1:
        return next(iter(visible)) + 1, "single_pose"
    if mixed == "drop":
        return None, "mixed_pose"
    # majority pose, ties dropped
    (top, n1), (_, n2) = visible.most_common(2)
    if n1 == n2:
        return None, "mixed_pose_tie"
    return top + 1, "mixed_majority"


def crop(image: Image.Image, t: Tile) -> Image.Image:
    region = image.crop((t.x, t.y, t.x + t.size, t.y + t.size))  # pads past edges
    return region.convert("RGB")


def write_tiles(tiles: list[Tile], out: Path, quality: int, deadline: float) -> bool:
    """Write missing tiles. Returns False if the deadline passed first; running
    the script again with the same arguments continues where it stopped."""
    by_source: dict[Path, list[Tile]] = {}
    for t in tiles:
        if not (out / t.name).exists():
            by_source.setdefault(t.source, []).append(t)
    for n, (source, group) in enumerate(sorted(by_source.items()), 1):
        with Image.open(source) as image:
            image.load()
            for t in group:
                path = out / t.name
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".part")
                crop(image, t).save(tmp, format="JPEG", quality=quality)
                tmp.replace(path)
        if n % 500 == 0:
            print(f"  wrote tiles for {n}/{len(by_source)} source images")
        if deadline and time.monotonic() > deadline:
            print(
                f"stopped after {n}/{len(by_source)} source images; "
                "run again to continue"
            )
            return False
    return True


def tile_name(t: Tile, root: Path) -> str:
    rel = t.source.relative_to(root).with_suffix("")
    safe = "__".join(rel.parts).replace(" ", "_")
    return f"tiles/{t.origin}/{safe}__{t.x}_{t.y}.jpg"


def snippet(args: argparse.Namespace, n_tiles: int) -> str:
    return f"""# Paste into your Hawk mission config (paths are examples: adjust them).
label-mode: script            # scripted labeler answers from the groundtruth
dataset:
  type: random
  class_list: [{", ".join(POSES)}]
  # stream_path is read on home by `hawk_deploy split`
  stream_path: {(args.out / "index" / "stream.txt").resolve()}
  index_path: index/scout.txt  # on each scout, relative to the scout data_root
  tiles_per_frame: {args.tiles_per_frame}
train_strategy:
  type: dnn_classifier
  bootstrap_path: {(args.out / "bootstrap.zip").resolve()}
selector:
  type: topk
  topk:
    k: 4
    batchSize: 1000
# stream: {n_tiles} tiles, positive rate target {args.positive_rate}
"""


def parse_pose_mix(text: str) -> dict[int, float]:
    shares: dict[int, float] = {}
    for item in text.split(","):
        name, _, value = item.partition("=")
        name = name.strip().lower()
        if name not in POSES:
            msg = f"unknown pose {name!r} in --pose-mix (poses: {', '.join(POSES)})"
            raise SystemExit(msg)
        shares[POSES.index(name) + 1] = float(value)
    total = sum(shares.values())
    if total <= 0:
        msg = "--pose-mix shares must add up to more than 0"
        raise SystemExit(msg)
    return {label: share / total for label, share in shares.items()}


def apply_pose_mix(positives: list[Tile], shares: dict[int, float]) -> list[Tile]:
    """Keep the largest number of positives that matches the requested shares.
    Poses left out of the mix are dropped."""
    by_label: dict[int, list[Tile]] = {}
    for t in positives:
        by_label.setdefault(t.label, []).append(t)
    total = min(
        len(by_label.get(label, [])) / share
        for label, share in shares.items()
        if share > 0
    )
    kept: list[Tile] = []
    for label, share in shares.items():
        kept += by_label.get(label, [])[: int(total * share)]
    return kept


def checksum(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, md5: str | None) -> None:
    """Download url to dest (via dest.part) and check its MD5."""
    part = dest.with_name(dest.name + ".part")
    print(f"downloading {url}\n         -> {dest}")
    with urllib.request.urlopen(url) as response, part.open("wb") as fp:  # noqa: S310
        if "text/html" in response.headers.get("Content-Type", ""):
            msg = "got a web page instead of a zip (the host wants a browser)"
            raise OSError(msg)
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        next_report = 0.0
        for chunk in iter(lambda: response.read(1 << 20), b""):
            fp.write(chunk)
            done += len(chunk)
            if total and done / total >= next_report:
                print(f"  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB")
                next_report += 0.1
    if md5 and checksum(part, "md5") != md5:
        part.unlink()
        msg = f"MD5 of {dest.name} does not match {md5}"
        raise OSError(msg)
    part.replace(dest)


def add_download_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dest", type=Path, default=Path("sar_data"))
    ap.add_argument("--aider-url", default=AIDER_URL)
    ap.add_argument("--c2a-url", default=C2A_URL)
    ap.add_argument("--no-verify", action="store_true", help="skip the MD5 checks")


def download_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="build_sar_mission.py download",
        description="Download AIDER (Zenodo) and C2A (Google Drive).",
    )
    add_download_args(ap)
    args = ap.parse_args(argv)
    args.dest.mkdir(parents=True, exist_ok=True)
    failed = False
    for name, url, md5 in (
        (AIDER_ZIP, args.aider_url, AIDER_MD5),
        (C2A_ZIP, args.c2a_url, C2A_MD5),
    ):
        path = args.dest / name
        expected = None if args.no_verify else md5
        if path.exists():
            if expected and checksum(path, "md5") != expected:
                print(
                    f"error: {path} exists but its MD5 is not {expected}",
                    file=sys.stderr,
                )
                failed = True
            else:
                print(f"have {path}")
            continue
        try:
            fetch(url, path, expected)
        except OSError as e:
            failed = True
            print(f"error: could not download {name}: {e}", file=sys.stderr)
            if name == C2A_ZIP:
                print(C2A_MANUAL.format(path=path), file=sys.stderr)
    return 1 if failed else 0


def extract_members(zip_path: Path, dest: Path, prefixes: list[str]) -> int:
    """Extract the members under the given prefixes; skip files already there."""
    dest = dest.resolve()
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not any(info.filename.startswith(p) for p in prefixes):
                continue
            out = (dest / info.filename).resolve()
            if dest not in out.parents:  # refuse paths that escape dest
                msg = f"unsafe path in {zip_path}: {info.filename}"
                raise SystemExit(msg)
            if out.exists() and out.stat().st_size == info.file_size:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            part = out.with_name(out.name + ".part")
            with zf.open(info) as src, part.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            part.replace(out)
            n += 1
    return n


def extract_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="build_sar_mission.py extract",
        description="Extract only the parts of both zips we use.",
    )
    ap.add_argument(
        "--dest",
        type=Path,
        default=Path("sar_data"),
        help="folder with the zips; images go to DEST/raw",
    )
    ap.add_argument("--aider-zip", type=Path, help=f"default: DEST/{AIDER_ZIP}")
    ap.add_argument("--c2a-zip", type=Path, help=f"default: DEST/{C2A_ZIP}")
    args = ap.parse_args(argv)
    raw = args.dest / "raw"
    for zip_path, members in (
        (args.aider_zip or args.dest / AIDER_ZIP, AIDER_MEMBERS),
        (args.c2a_zip or args.dest / C2A_ZIP, C2A_MEMBERS),
    ):
        if not zip_path.exists():
            print(
                f"error: missing {zip_path}; run the download step first",
                file=sys.stderr,
            )
            return 1
        n = extract_members(zip_path, raw, members)
        print(f"{zip_path.name}: extracted {n} new files into {raw}")
    return 0


def validate_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="build_sar_mission.py validate",
        description="Check a built mission folder. With --write-manifest, record "
        "a SHA-256 for every file; otherwise check the files against an existing "
        "manifest.",
    )
    ap.add_argument("--out", type=Path, required=True, help="the folder `build` wrote")
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args(argv)
    out = args.out
    errors: list[str] = []

    stats = json.loads((out / "stats.json").read_text())
    size = int(stats["args"]["tile_size"])
    lines = (out / "index" / "stream.txt").read_text().splitlines()
    counts: Counter[int] = Counter()
    stream_sources = set()
    for line in lines:
        path, label = line.rsplit(maxsplit=1)
        if not label.isdigit() or int(label) > len(POSES):
            errors.append(f"bad label: {line}")
            continue
        counts[int(label)] += 1
        stream_sources.add(Path(path).name.rsplit("__", 1)[0])
        try:
            with Image.open(out / path) as im:
                if im.size != (size, size):
                    errors.append(f"{path} is {im.size}, expected {size}x{size}")
        except OSError as e:
            errors.append(f"{path}: {e}")
    expected = stats["stream_by_class"]
    got = {"negative": counts[0], **{p: counts[i + 1] for i, p in enumerate(POSES)}}
    if got != expected:
        errors.append(f"stream class counts {got} differ from stats.json {expected}")

    with zipfile.ZipFile(out / "bootstrap.zip") as zf:
        names = zf.namelist()
    if {Path(n).parts[0] for n in names} - {str(i) for i in range(len(POSES) + 1)}:
        errors.append("bootstrap.zip has folders other than 0..5")
    if {Path(n).name.rsplit("__", 1)[0] for n in names} & stream_sources:
        errors.append("a bootstrap source image also appears in the stream")

    manifest = out / "manifest.sha256"
    files = ["index/stream.txt", "bootstrap.zip"] + [
        line.rsplit(maxsplit=1)[0] for line in lines
    ]
    if args.write_manifest:
        manifest.write_text("".join(f"{checksum(out / f)}  {f}\n" for f in files))
        print(f"wrote {manifest} ({len(files)} files)")
    elif manifest.exists():
        recorded = dict(
            reversed(line.split("  ", 1)) for line in manifest.read_text().splitlines()
        )
        for f in files:
            if recorded.get(f) != checksum(out / f):
                errors.append(f"checksum differs: {f}")

    print(f"stream: {len(lines)} tiles {got}")
    print(f"stream.txt sha256: {checksum(out / 'index' / 'stream.txt')}")
    for e in errors[:20]:
        print("error:", e, file=sys.stderr)
    if errors:
        print(f"{len(errors)} problems found", file=sys.stderr)
        return 1
    print("ok")
    return 0


def build_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="build_sar_mission.py build",
        description="Tile, label and merge the two datasets.",
    )
    ap.add_argument(
        "--c2a-root",
        type=Path,
        required=True,
        help="C2A folder that holds train/ val/ test/ and "
        "'All labels with Pose information/'",
    )
    ap.add_argument(
        "--aider-root",
        type=Path,
        required=True,
        help="extracted AIDER folder (one subfolder per class)",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--aider-classes",
        default="normal",
        help="comma list of AIDER class folders to use as negatives (default: normal)",
    )
    ap.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    ap.add_argument(
        "--min-visible",
        type=float,
        default=0.5,
        help="fraction of a person's box that must be inside a tile to count",
    )
    ap.add_argument(
        "--crop-margin",
        type=int,
        help="pixels at each tile edge that Hawk's classifier crops away "
        "(ResNet resizes to 256 and crops 224); people are counted only "
        "inside it (default: tile size / 16, 0 to count the whole tile)",
    )
    ap.add_argument(
        "--mixed",
        choices=["drop", "majority"],
        default="drop",
        help="what to do with a tile that holds people of different poses",
    )
    ap.add_argument(
        "--positive-rate",
        type=float,
        default=0.05,
        help="target share of tiles with a person in the stream",
    )
    ap.add_argument(
        "--pose-mix",
        default=DEFAULT_POSE_MIX,
        help="relative share of each pose among the positives, e.g. "
        "'upright=0.4,sitting=0.25,bent=0.15,kneeling=0.15,lying=0.05'; "
        "C2A's poses are roughly balanced, so this is how a pose is made rare "
        "('natural' keeps C2A's own mix; default: our mission_v1 mix)",
    )
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=0,
        help="stop writing tiles after this many seconds (0 = no limit); "
        "rerun the same command to continue",
    )
    ap.add_argument(
        "--no-c2a-empty-negatives",
        dest="c2a_empty_negatives",
        action="store_false",
        help="use only AIDER tiles as negatives, not the empty C2A tiles",
    )
    ap.add_argument(
        "--limit-images",
        type=int,
        default=0,
        help="use at most this many C2A images (0 = all); for a quick subset",
    )
    ap.add_argument("--bootstrap-per-pose", type=int, default=20)
    ap.add_argument("--bootstrap-negatives", type=int, default=100)
    ap.add_argument("--tiles-per-frame", type=int, default=200)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="compute labels and stats only, write no images",
    )
    args = ap.parse_args(argv)

    if not 0 < args.positive_rate < 1:
        ap.error("--positive-rate must be between 0 and 1")
    if args.crop_margin is None:
        args.crop_margin = args.tile_size // 16
    if not 0 <= args.crop_margin < args.tile_size / 2:
        ap.error("--crop-margin must be at least 0 and less than half a tile")
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
                label, why = label_tile(
                    boxes,
                    x,
                    y,
                    args.tile_size,
                    args.min_visible,
                    args.mixed,
                    args.crop_margin,
                )
                reasons[why] += 1
                if label is not None:
                    c2a_tiles.append(
                        Tile(image_path, "c2a", split, x, y, args.tile_size, label)
                    )

    # 2. AIDER tiles are all negatives
    aider_images = find_aider_images(
        args.aider_root, [c.strip() for c in args.aider_classes.split(",")]
    )
    aider_tiles: list[Tile] = []
    for image_path in aider_images:
        with Image.open(image_path) as im:
            width, height = im.size
        for y in grid(height, args.tile_size):
            for x in grid(width, args.tile_size):
                aider_tiles.append(
                    Tile(image_path, "aider", "aider", x, y, args.tile_size, 0)
                )

    # 3. bootstrap: from C2A train images and some AIDER images; every tile of
    #    a bootstrap source image is kept out of the stream
    boot: list[Tile] = []
    by_pose: dict[int, list[Tile]] = {}
    for t in c2a_tiles:
        if t.split == "train" and t.label > 0:
            by_pose.setdefault(t.label, []).append(t)
    for label in range(1, len(POSES) + 1):
        pool = by_pose.get(label, [])
        rng.shuffle(pool)
        boot += pool[: args.bootstrap_per_pose]
        if len(pool) < args.bootstrap_per_pose:
            print(
                f"warning: only {len(pool)} bootstrap tiles for {POSES[label - 1]}",
                file=sys.stderr,
            )
    # negatives from AIDER and, when the stream has them, from empty C2A train
    # tiles, in the same proportion as the stream negatives; with AIDER alone
    # the model learns that any C2A tile is a person
    neg_pool = aider_tiles[:]
    rng.shuffle(neg_pool)
    num_aider = args.bootstrap_negatives
    if args.c2a_empty_negatives:
        c2a_empty = [t for t in c2a_tiles if t.label == 0]
        c2a_share = len(c2a_empty) / max(1, len(c2a_empty) + len(aider_tiles))
        c2a_pool = [t for t in c2a_empty if t.split == "train"]
        rng.shuffle(c2a_pool)
        num_c2a = round(args.bootstrap_negatives * c2a_share)
        boot += c2a_pool[:num_c2a]
        num_aider -= num_c2a
    boot += neg_pool[:num_aider]
    held_out = {t.source for t in boot}

    # 4. stream: all remaining positives, plus negatives up to the target rate
    positives = [t for t in c2a_tiles if t.label > 0 and t.source not in held_out]
    negatives = [t for t in aider_tiles if t.source not in held_out]
    if args.c2a_empty_negatives:
        negatives += [t for t in c2a_tiles if t.label == 0 and t.source not in held_out]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    if args.pose_mix and args.pose_mix != "natural":
        positives = apply_pose_mix(positives, parse_pose_mix(args.pose_mix))
        rng.shuffle(positives)
    want_neg = round(len(positives) * (1 - args.positive_rate) / args.positive_rate)
    if want_neg > len(negatives):
        # not enough negatives: keep all of them and drop positives instead
        keep_pos = round(len(negatives) * args.positive_rate / (1 - args.positive_rate))
        print(
            f"note: only {len(negatives)} negative tiles; keeping {keep_pos} of "
            f"{len(positives)} positive tiles to reach the target rate",
            file=sys.stderr,
        )
        positives = positives[:keep_pos]
        want_neg = len(negatives)
    stream = positives + negatives[:want_neg]
    rng.shuffle(stream)

    for t in boot + stream:
        root = args.c2a_root if t.origin == "c2a" else args.aider_root
        t.name = tile_name(t, root)

    # 5. stats
    def per_class(tiles: list[Tile]) -> dict[str, int]:
        c = Counter(t.label for t in tiles)
        return {"negative": c[0], **{p: c[i + 1] for i, p in enumerate(POSES)}}

    stats.update(
        {
            "c2a_images": len(c2a),
            "c2a_people_per_image_mean": round(
                sum(people_per_image) / max(1, len(people_per_image)), 1
            ),
            "aider_images": len(aider_images),
            "c2a_tile_outcomes": dict(reasons),
            "c2a_tiles_by_class": per_class(c2a_tiles),
            "aider_tiles": len(aider_tiles),
            "bootstrap_by_class": per_class(boot),
            "stream_by_class": per_class(stream),
            "stream_tiles": len(stream),
            "stream_positive_rate": round(
                sum(t.label > 0 for t in stream) / max(1, len(stream)), 4
            ),
        }
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "args"}, indent=2))
    if args.dry_run:
        return 0

    # 6. write tiles, stream, bootstrap, config snippet
    print(f"writing {len(boot) + len(stream)} tiles to {args.out}/tiles ...")
    deadline = time.monotonic() + args.max_seconds if args.max_seconds else 0.0
    if not write_tiles(boot + stream, args.out, args.quality, deadline):
        return 3
    index_dir = args.out / "index"
    index_dir.mkdir(exist_ok=True)
    (index_dir / "stream.txt").write_text(
        "".join(f"{t.name} {t.label}\n" for t in stream)
    )
    with zipfile.ZipFile(args.out / "bootstrap.zip", "w") as zf:
        for t in boot:
            zf.write(args.out / t.name, f"{t.label}/{Path(t.name).name}")
    (args.out / "mission_snippet.yml").write_text(snippet(args, len(stream)))
    stats["stream_sha256"] = checksum(index_dir / "stream.txt")
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print("done:", args.out / "index" / "stream.txt", args.out / "bootstrap.zip")
    return 0


def all_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="build_sar_mission.py all",
        description="Download, extract, build and validate, with our settings.",
    )
    add_download_args(ap)
    ap.add_argument("--out", type=Path, help="default: DEST/mission_v1")
    args = ap.parse_args(argv)
    dest, raw = str(args.dest), args.dest / "raw"
    out = str(args.out or args.dest / "mission_v1")
    download = [
        "--dest",
        dest,
        "--aider-url",
        args.aider_url,
        "--c2a-url",
        args.c2a_url,
    ]
    if args.no_verify:
        download.append("--no-verify")
    build = [
        "--c2a-root",
        str(raw / "new_dataset3"),
        "--aider-root",
        str(raw / "AIDER"),
    ]
    steps = [
        (download_main, download),
        (extract_main, ["--dest", dest]),
        (build_main, [*build, "--out", out]),
        (validate_main, ["--out", out, "--write-manifest"]),
    ]
    for step, step_argv in steps:
        rc = step(step_argv)
        if rc:
            return rc
    return 0


STEPS = {
    "download": download_main,
    "extract": extract_main,
    "build": build_main,
    "validate": validate_main,
    "all": all_main,
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in STEPS:
        return STEPS[argv[0]](argv[1:])
    print(__doc__)
    print("steps:", ", ".join(STEPS))
    return 0 if argv[:1] in (["-h"], ["--help"]) else 2


if __name__ == "__main__":
    sys.exit(main())
