# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for build_sar_mission.py on small fake C2A / AIDER folders laid out
like the real downloads."""

from __future__ import annotations

import json
import random
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import build_sar_mission as bsm  # noqa: E402


def _img(path: Path, w: int, h: int, color=(120, 110, 90)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), color).save(path)


def _yolo(
    path: Path, w: int, h: int, boxes: list[tuple[int, int, int, int, int]]
) -> None:
    """boxes are pixel (x0, y0, x1, y1, pose)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for x0, y0, x1, y1, pose in boxes:
        lines.append(
            f"0 {(x0 + x1) / 2 / w} {(y0 + y1) / 2 / h} "
            f"{(x1 - x0) / w} {(y1 - y0) / h} {pose}",
        )
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def fake(tmp_path: Path) -> tuple[Path, Path]:
    rng = random.Random(1)
    c2a = tmp_path / "new_dataset3"
    pose_dir = c2a / bsm.C2A_POSE_LABEL_DIRS[0]
    n = 0
    for split, count in (("train", 40), ("val", 10), ("test", 10)):
        for _ in range(count):
            n += 1
            w, h = rng.choice([(512, 512), (600, 420), (200, 150)])
            _img(c2a / split / "images" / f"img{n:04}.jpg", w, h)
            boxes = []
            for _ in range(rng.randint(1, 6)):
                x0, y0 = rng.randint(0, w - 30), rng.randint(0, h - 30)
                boxes.append((x0, y0, x0 + 20, y0 + 25, rng.randint(0, 4)))
            _yolo(pose_dir / f"img{n:04}.txt", w, h, boxes)
    aider = tmp_path / "AIDER" / "AIDER"
    for cls, count in (("normal", 120), ("fire", 5), ("collapsed_building", 5)):
        for i in range(count):
            _img(aider / cls / f"{cls}_{i:04}.jpg", 240, 240, (60, 140, 60))
    return c2a, tmp_path / "AIDER"


def test_grid() -> None:
    assert bsm.grid(100, 256) == [0]
    assert bsm.grid(512, 256) == [0, 256]
    assert bsm.grid(600, 256) == [0, 256, 344]  # last tile stays inside


def test_label_tile_rules() -> None:
    upright = bsm.Box(10, 10, 30, 40, 4)
    lying = bsm.Box(50, 50, 80, 60, 2)
    cut = bsm.Box(250, 10, 270, 40, 4)  # mostly outside a 256 tile at 0,0
    assert bsm.label_tile([], 0, 0, 256, 0.5, "drop") == (0, "empty")
    assert bsm.label_tile([upright], 0, 0, 256, 0.5, "drop") == (5, "single_pose")
    assert bsm.label_tile([upright, lying], 0, 0, 256, 0.5, "drop") == (
        None,
        "mixed_pose",
    )
    assert bsm.label_tile([upright, upright, lying], 0, 0, 256, 0.5, "majority") == (
        5,
        "mixed_majority",
    )
    assert bsm.label_tile([upright, lying], 0, 0, 256, 0.5, "majority") == (
        None,
        "mixed_pose_tie",
    )
    assert bsm.label_tile([cut], 0, 0, 256, 0.5, "drop") == (None, "partial_only")
    # a cut-off person next to a full one does not change the label
    assert bsm.label_tile([upright, cut], 0, 0, 256, 0.5, "drop") == (5, "single_pose")


def test_read_c2a_boxes(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    _yolo(f, 400, 200, [(100, 50, 140, 100, 2)])
    (b,) = bsm.read_c2a_boxes(f, 400, 200)
    assert (round(b.x0), round(b.y0), round(b.x1), round(b.y1), b.pose) == (
        100,
        50,
        140,
        100,
        2,
    )
    f.write_text("0 0.5 0.5 0.1 0.1\n")  # no pose field
    with pytest.raises(ValueError):
        bsm.read_c2a_boxes(f, 400, 200)


def test_end_to_end(fake: tuple[Path, Path], tmp_path: Path) -> None:
    c2a, aider = fake
    out = tmp_path / "out"
    rc = bsm.main(
        [
            "--c2a-root",
            str(c2a),
            "--aider-root",
            str(aider),
            "--out",
            str(out),
            "--positive-rate",
            "0.2",
            "--bootstrap-per-pose",
            "2",
            "--bootstrap-negatives",
            "5",
            "--seed",
            "3",
        ]
    )
    assert rc == 0
    stats = json.loads((out / "stats.json").read_text())

    # stream: every line points to a written tile, labels in range
    lines = (out / "index" / "stream.txt").read_text().splitlines()
    assert len(lines) == stats["stream_tiles"] > 0
    labels = []
    for line in lines:
        path, label = line.rsplit(maxsplit=1)
        assert (out / path).is_file()
        with Image.open(out / path) as im:
            assert im.size == (bsm.DEFAULT_TILE_SIZE, bsm.DEFAULT_TILE_SIZE)
        labels.append(int(label))
    assert set(labels) <= set(range(len(bsm.POSES) + 1))
    rate = sum(lbl > 0 for lbl in labels) / len(labels)
    assert abs(rate - 0.2) < 0.02

    # only AIDER "normal" folders are used
    assert not any("fire" in line or "collapsed" in line for line in lines)

    # bootstrap: folders 0..5 only, and no bootstrap source image in the stream
    with zipfile.ZipFile(out / "bootstrap.zip") as zf:
        names = zf.namelist()
    dirs = {Path(n).parts[0] for n in names}
    assert dirs <= {str(i) for i in range(6)} and "0" in dirs
    boot_sources = {Path(n).name.rsplit("__", 1)[0] for n in names}
    stream_sources = {
        Path(line.rsplit(maxsplit=1)[0]).name.rsplit("__", 1)[0] for line in lines
    }
    assert boot_sources and not boot_sources & stream_sources

    assert (out / "mission_snippet.yml").read_text().count("class_list") == 1


def test_stream_parses_with_hawk_random_retriever(
    fake: tuple[Path, Path], tmp_path: Path
) -> None:
    """The stream lines must turn into the right class names inside Hawk."""
    hawk = pytest.importorskip("hawk.scout.retrieval.random_retriever")
    from hawk.classes import ClassList, ClassName

    c2a, aider = fake
    out = tmp_path / "out"
    bsm.main(
        [
            "--c2a-root",
            str(c2a),
            "--aider-root",
            str(aider),
            "--out",
            str(out),
            "--positive-rate",
            "0.3",
            "--bootstrap-per-pose",
            "1",
            "--bootstrap-negatives",
            "2",
        ]
    )

    class Ctx:
        class_list = ClassList(bsm.POSES)

    r = hawk.RandomRetriever.__new__(hawk.RandomRetriever)
    r._context = Ctx()
    r._index_root = out
    r.config = type("C", (), {"data_root": out})()
    for line in (out / "index" / "stream.txt").read_text().splitlines()[:50]:
        path, label = line.rsplit(maxsplit=1)
        oid = r._tile_to_objectid(line).oid
        expected = "negative" if label == "0" else bsm.POSES[int(label) - 1]
        assert oid.startswith(f"/{ClassName(expected)}/collection/id/")
        assert oid.endswith(path)


def test_pose_mix_makes_a_pose_rare() -> None:
    tiles = [
        bsm.Tile(Path(f"s{i}"), "c2a", "train", 0, 0, 128, 1 + i % 5)
        for i in range(1000)
    ]
    shares = bsm.parse_pose_mix(
        "upright=0.4,sitting=0.3,bent=0.1,kneeling=0.15,lying=0.05"
    )
    kept = bsm.apply_pose_mix(tiles, shares)
    counts = {p: sum(t.label == i + 1 for t in kept) for i, p in enumerate(bsm.POSES)}
    assert (
        counts["upright"] == 200
    )  # the scarcest pose relative to its share sets the size
    assert counts["lying"] == 25
    assert counts["sitting"] == 150
    with pytest.raises(SystemExit):
        bsm.parse_pose_mix("flying=1")


def test_tile_writing_resumes(fake: tuple[Path, Path], tmp_path: Path) -> None:
    c2a, aider = fake
    out = tmp_path / "out"
    common = [
        "--c2a-root",
        str(c2a),
        "--aider-root",
        str(aider),
        "--out",
        str(out),
        "--positive-rate",
        "0.2",
        "--bootstrap-per-pose",
        "1",
        "--bootstrap-negatives",
        "2",
    ]
    assert bsm.main([*common, "--max-seconds", "1e-9"]) == 3  # stops right away
    assert not (out / "index" / "stream.txt").exists()
    assert bsm.main(common) == 0  # second run finishes
    lines = (out / "index" / "stream.txt").read_text().splitlines()
    assert all((out / line.rsplit(maxsplit=1)[0]).is_file() for line in lines)
    assert not list(out.rglob("*.part"))


# ------------------------------------------------- download, extract, validate


def _zip_tree(root: Path, top: str, zip_path: Path) -> None:
    """Zip root/ so its members start with top/, like the real archives."""
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in sorted(root.rglob("*")):
            if f.is_file():
                zf.write(f, f"{top}/{f.relative_to(root)}")


@pytest.fixture
def fake_zips(fake: tuple[Path, Path], tmp_path: Path) -> tuple[Path, Path]:
    c2a, aider = fake
    (c2a / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (c2a / "train" / "labels" / "unused.txt").write_text("not needed\n")
    zips = tmp_path / "zips"
    zips.mkdir()
    _zip_tree(c2a, "new_dataset3", zips / bsm.C2A_ZIP)
    _zip_tree(aider / "AIDER", "AIDER", zips / bsm.AIDER_ZIP)
    return zips / bsm.AIDER_ZIP, zips / bsm.C2A_ZIP


def test_all_steps_reproduce_and_validate(
    fake_zips: tuple[Path, Path], tmp_path: Path
) -> None:
    aider_zip, c2a_zip = fake_zips
    dest = tmp_path / "sar_data"
    rc = bsm.main(
        [
            "all",
            "--dest",
            str(dest),
            "--no-verify",
            "--aider-url",
            aider_zip.as_uri(),
            "--c2a-url",
            c2a_zip.as_uri(),
        ]
    )
    assert rc == 0
    out = dest / "mission_v1"
    assert (out / "manifest.sha256").is_file()
    stats = json.loads((out / "stats.json").read_text())
    assert stats["stream_sha256"] == bsm.sha256sum(out / "index" / "stream.txt")
    # only the parts we need were extracted
    assert not (dest / "raw" / "new_dataset3" / "train" / "labels").exists()
    assert not (dest / "raw" / "AIDER" / "fire").exists()

    # same inputs and seed give the same stream
    again = tmp_path / "again"
    bsm.main(
        [
            "build",
            "--c2a-root",
            str(dest / "raw" / "new_dataset3"),
            "--aider-root",
            str(dest / "raw" / "AIDER"),
            "--out",
            str(again),
        ]
    )
    assert (again / "index" / "stream.txt").read_text() == (
        out / "index" / "stream.txt"
    ).read_text()

    # validate passes, then catches a changed tile
    assert bsm.main(["validate", "--out", str(out)]) == 0
    first_tile = (out / "index" / "stream.txt").read_text().split()[0]
    Image.new("RGB", (bsm.DEFAULT_TILE_SIZE,) * 2, (255, 0, 0)).save(out / first_tile)
    assert bsm.main(["validate", "--out", str(out)]) == 1


def test_validate_catches_missing_tile_and_wrong_size(
    fake: tuple[Path, Path], tmp_path: Path
) -> None:
    c2a, aider = fake
    out = tmp_path / "out"
    bsm.main(
        [
            "build",
            "--c2a-root",
            str(c2a),
            "--aider-root",
            str(aider),
            "--out",
            str(out),
            "--bootstrap-per-pose",
            "1",
            "--bootstrap-negatives",
            "2",
        ]
    )
    lines = (out / "index" / "stream.txt").read_text().split("\n")
    (out / lines[0].rsplit(maxsplit=1)[0]).unlink()
    Image.new("RGB", (10, 10)).save(out / lines[1].rsplit(maxsplit=1)[0])
    assert bsm.main(["validate", "--out", str(out)]) == 1


def test_download_rejects_bad_checksum(
    fake_zips: tuple[Path, Path], tmp_path: Path
) -> None:
    aider_zip, c2a_zip = fake_zips
    dest = tmp_path / "dl"
    rc = bsm.main(
        [
            "download",
            "--dest",
            str(dest),
            "--aider-url",
            aider_zip.as_uri(),
            "--c2a-url",
            c2a_zip.as_uri(),
        ]
    )
    assert rc == 1  # the fake zips do not match the real MD5 sums
    assert not (dest / bsm.AIDER_ZIP).exists()
    assert not list(dest.glob("*.part"))


def test_extract_refuses_paths_outside_dest(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("AIDER/normal/../../../escape.jpg", b"x")
    with pytest.raises(SystemExit):
        bsm.extract_members(bad, tmp_path / "raw", bsm.AIDER_MEMBERS)
    assert not (tmp_path / "escape.jpg").exists()
