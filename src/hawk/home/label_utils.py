# SPDX-FileCopyrightText: 2022-2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Utility functions for labelers"""

from __future__ import annotations

import json
import time
from dataclasses import InitVar, asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, TextIO

from logzero import logger

from .utils import tailf

if TYPE_CHECKING:
    from collections.abc import Container
    from os import PathLike


@dataclass
class ClassMap:
    class_map: dict[int, str]
    inverse_map: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.inverse_map = {name: label for label, name in self.class_map.items()}

    @classmethod
    def from_list(cls, classes: list[str]) -> ClassMap:
        """Create a mapping from a list by applying incrementing label numbers"""
        return cls(class_map=dict(enumerate(classes)))

    @property
    def classes(self) -> list[str]:
        return list(self.inverse_map)

    def __getitem__(self, item: int | str) -> str:
        """Tries to accept either labels or names and maps them to class names
        Will create new labels for previously unseen class names (or labels).
        """
        try:
            class_label = int(item)
        except ValueError:
            class_label = self.name_to_label(str(item))
        return self.label_to_name(class_label)

    def __setitem__(self, label: int, name: str) -> None:
        """Adds a new class label <> name mapping."""
        assert label not in self.class_map
        assert name not in self.inverse_map
        self.class_map[label] = name
        self.inverse_map[name] = label

    def name_to_label(self, class_name: str) -> int:
        """Returns label for the given class.
        Creates a new label for the class if it did not previously exist.
        """
        try:
            return self.inverse_map[class_name]
        except ValueError:
            new_label = max(self.class_map) + 1
            logger.warning(f"Adding unknown class {class_name} with label {new_label}")
            self[new_label] = class_name
            return new_label

    def label_to_name(self, class_label: int) -> str:
        """Given a label, return the associated class name."""
        try:
            return self.class_map[class_label]
        except IndexError:
            class_name = str(class_label)
            logger.warning(f"Adding unknown class label {class_label}")
            self[class_label] = class_name
            return class_name


@dataclass(order=True)
class Detection:
    # canonical min/max coordinates which are easier to draw in browser
    minX: float = 0.0
    minY: float = 0.0
    maxX: float = 1.0
    maxY: float = 1.0

    cls_scores: dict[str, float] = field(default_factory=dict)

    # center x/y/width/height output from yolo (and easier to cluster?)
    x: InitVar[float | None] = None
    y: InitVar[float | None] = None
    w: InitVar[float] = 1.0
    h: InitVar[float] = 1.0

    def __post_init__(
        self, x: float | None, y: float | None, w: float, h: float
    ) -> None:
        if x is not None:
            w /= 2
            self.minX = x - w
            self.maxX = x + w
        if y is not None:
            h /= 2
            self.minY = y - h
            self.maxY = y + h

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> Detection:
        # filter out negatives (and 0 scores)
        cls_scores = {
            cls: score
            for cls, score in obj.pop("cls_scores", {}).items()
            if score and cls not in ["", "neg", "negative"]
        }
        return cls(cls_scores=cls_scores, **obj)

    @classmethod
    def from_yolo(cls, line: str, class_map: ClassMap) -> Detection:
        label, centerX, centerY, width, height, *_score = line.split()
        try:
            class_name = class_map[int(label)]  # + 1]
        except ValueError:
            class_name = class_map[label]
        score = float(_score[0]) if _score else 1.0
        return cls(
            cls_scores={class_name: score},
            x=float(centerX),
            y=float(centerY),
            w=float(width),
            h=float(height),
        )

    def to_yolo(
        self, class_map: ClassMap | None = None, with_scores: bool = False
    ) -> Iterator[str]:
        """Yolo, using only positive classes counting from 0
        If no class_map has been given, will use class names instead.
        """
        centerX = (self.maxX + self.minX) / 2
        centerY = (self.maxY + self.minY) / 2
        deltaX = self.maxX - self.minX
        deltaY = self.maxY - self.minY
        for cls, score in self.cls_scores.items():
            label = cls if class_map is None else class_map.name_to_label(cls)  # - 1
            score_str = f" {score}" if with_scores else ""
            yield f"{label} {centerX} {centerY} {deltaX} {deltaY}{score_str}"

    @classmethod
    def merge_detections(cls, detections: Iterator[Detection]) -> Iterator[Detection]:
        prev: Detection | None = None

        # relies on the fact that the coordinates come before the cls_scores...
        for cur in sorted(detections):
            if prev is None:
                prev = cur
                continue

            if prev.coords == cur.coords:
                prev.cls_scores.update(cur.cls_scores)
                continue

            yield prev
            prev = cur

        if prev is not None:
            yield prev

    @property
    def classes(self) -> set[str]:
        return {cls for cls in self.cls_scores}

    @property
    def coords(self) -> tuple[float, float, float, float]:
        return self.minX, self.minY, self.maxX, self.maxY

    @property
    def max_score(self) -> float:
        return max(self.cls_scores.values())


@dataclass
class LabelSample:
    """Representation of an unlabeled tile received from a scout on it's way
    to getting labeled, or a labeled result being passed back to the scout"""

    objectId: str  # unique object id
    scoutIndex: int  # index of originating scout
    queued: float = field(default_factory=time.time)
    detections: list[Detection] = field(default_factory=list)
    line: InitVar[int] = -1  # used to track line number in jsonl file

    def __post_init__(self, line: int) -> None:
        self.index = line

    def to_jsonl(self, fp: TextIO) -> None:
        jsonl = json.dumps(
            dict(
                objectId=self.objectId,
                scoutIndex=self.scoutIndex,
                queued=self.queued,
                detections=[asdict(detection) for detection in self.detections],
            )
        )
        fp.write(f"{jsonl}\n")

    def to_yolo(self, class_map: ClassMap | None = None) -> str:
        """Yolo using class labels that start counting at 0
        Will use class names when class_map is None
        """
        return "".join(
            line + "\n"
            for detection in self.detections
            for line in detection.to_yolo(class_map)
        )

    @classmethod
    def from_dict(cls, obj: dict[str, Any], line: int = -1) -> LabelSample:
        detections = [
            Detection.from_dict(detection) for detection in obj.pop("detections", [])
        ]
        return cls(line=line, detections=detections, **obj)

    @property
    def classes(self) -> set[str]:
        return set.union(*[detection.classes for detection in self.detections])

    @property
    def max_score(self) -> float:
        return (
            max(detection.max_score for detection in self.detections)
            if self.detections
            else 0.0
        )


def index_jsonl(
    jsonl: PathLike[str] | str, skip: int = 0, index: set[str] | None = None
) -> tuple[set[str], int]:
    """Returns a set of all unique ids in the given jsonl file.
    Returns both the set and the number of lines parsed."""
    jsonl_path = Path(jsonl)
    if not jsonl_path.exists():
        return set(), 0

    if index is None:
        index = set()

    n = 0  # in case the file is empty
    with jsonl_path.open() as fp:
        for n, line in enumerate(fp, start=1):
            if n <= skip:
                continue
            obj = json.loads(line)
            index.add(obj["objectId"])
    return index, n


def read_jsonl(
    jsonl: PathLike[str] | str,
    exclude: Container[str] | None = None,
    skip: int = 0,
    tail: bool = False,
) -> Iterator[LabelSample]:
    """Yields all entries in the given jsonl file
    exclude is a set of id values to leave out
    skip is the number of lines to skip completely before parsing lines
    when tail is true, iterate indefinitely and block waiting for new lines
    """
    if exclude is None:
        exclude = set()

    jsonl_path = Path(jsonl)
    while not jsonl_path.exists():
        if not tail:
            return
        # wait for file
        time.sleep(0.5)

    with jsonl_path.open() as fp:
        line_iter = tailf(fp) if tail else fp
        for index, line in enumerate(line_iter, start=1):
            if index <= skip:
                continue
            obj = json.loads(line)
            objectId = obj["objectId"]
            if objectId in exclude:
                logger.debug(f"Not loading previously labeled {objectId}")
                continue
            yield LabelSample.from_dict(obj, index)


@dataclass
class MissionResults:
    mission_dir: PathLike[str] | str = field(default_factory=Path.cwd)

    labeled_jsonl: Path = field(init=False, repr=False)
    unlabeled_jsonl: Path = field(init=False, repr=False)

    labeled: dict[str, int] = field(default_factory=dict)
    labeled_offset: int = 0

    unlabeled: list[LabelSample] = field(default_factory=list)
    unlabeled_offset: int = 0

    def __post_init__(self) -> None:
        self.labeled_jsonl = Path(self.mission_dir, "labeled.jsonl")
        self.unlabeled_jsonl = Path(self.mission_dir, "unlabeled.jsonl")

    def resync_labeled(self) -> None:
        new_labels = list(read_jsonl(self.labeled_jsonl, skip=self.labeled_offset))
        if new_labels:
            self.labeled.update(
                (label.objectId, len(label.detections)) for label in new_labels
            )
            self.labeled_offset = new_labels[-1].index

    def resync(self) -> None:
        self.resync_labeled()

        new_unlabeled = list(
            read_jsonl(self.unlabeled_jsonl, skip=self.unlabeled_offset)
        )
        if new_unlabeled:
            self.unlabeled.extend(new_unlabeled)
            self.unlabeled_offset = new_unlabeled[-1].index

    @property
    def classes(self) -> set[str]:
        return self.unlabeled[-1].classes if self.unlabeled else set()

    def read_unlabeled(
        self, exclude_labeled: bool = True, tail: bool = False
    ) -> Iterator[LabelSample]:
        if exclude_labeled:
            self.resync_labeled()
            exclude: Container[str] = self.labeled
        else:
            exclude = set()
        yield from read_jsonl(self.unlabeled_jsonl, exclude=exclude, tail=tail)

    def save_labeled(self, results: list[LabelSample]) -> None:
        # make sure we're current with the on-disk labeled state
        self.resync_labeled()

        with self.labeled_jsonl.open("a") as fp:
            for result in results:
                # skip already labeled results
                if result.objectId in self.labeled:
                    logger.debug(f"Not saving previously labeled {result.objectId}")
                    continue

                # skip if there are still unclassified bounding boxes?
                for detection in result.detections:
                    for cls, score in detection.cls_scores.items():
                        if not cls or score != 1.0:
                            continue

                result.to_jsonl(fp)
