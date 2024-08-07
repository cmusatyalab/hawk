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


@dataclass
class BoundingBox:
    label: str
    score: float = 1.0
    minX: float = 0.0
    minY: float = 0.0
    maxX: float = 1.0
    maxY: float = 1.0

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> BoundingBox:
        return cls(**obj)

    @classmethod
    def from_yolo(cls, line: str, class_map: ClassMap) -> BoundingBox:
        label, centerX_, centerY_, width, height, *score = line.split()
        centerX, centerY = float(centerX_), float(centerY_)
        deltaX, deltaY = float(width) / 2, float(height) / 2
        try:
            class_name = class_map[int(label) + 1]
        except ValueError:
            class_name = class_map[label]
        return cls(
            label=class_name,
            minX=centerX - deltaX,
            minY=centerY - deltaY,
            maxX=centerX + deltaX,
            maxY=centerY + deltaY,
            score=float(score[0]) if score else 1.0,
        )

    def to_yolo(self, class_map: ClassMap | None = None) -> str:
        """Yolo, using only positive classes counting from 0
        If no class_map has been given, will use class names instead.
        """
        centerX = (self.maxX + self.minX) / 2
        centerY = (self.maxY + self.minY) / 2
        deltaX = self.maxX - self.minX
        deltaY = self.maxY - self.minY
        label = (
            self.label if class_map is None else class_map.name_to_label(self.label) - 1
        )
        return f"{label} {centerX} {centerY} {deltaX} {deltaY} {self.score}"


@dataclass
class LabelSample:
    """Representation of an unlabeled tile received from a scout on it's way
    to getting labeled, or a labeled result being passed back to the scout"""

    objectId: str  # unique object id
    scoutIndex: int  # index of originating scout
    queued: float = field(default_factory=time.time)
    labels: list[BoundingBox] = field(default_factory=list)
    line: InitVar[int] = -1  # used to track line number in jsonl file

    def __post_init__(self, line: int) -> None:
        self.index = line

    @property
    def score(self) -> float:
        return max(bbox.score for bbox in self.labels) if self.labels else 0.0

    def to_jsonl(self, fp: TextIO) -> None:
        jsonl = json.dumps(
            dict(
                objectId=self.objectId,
                scoutIndex=self.scoutIndex,
                queued=self.queued,
                labels=[asdict(bbox) for bbox in self.labels],
            )
        )
        fp.write(f"{jsonl}\n")

    def to_yolo(self, class_map: ClassMap | None = None) -> str:
        """Yolo using class labels that start counting at 0
        Will use class names when class_map is None
        """
        return "".join(label.to_yolo(class_map) + "\n" for label in self.labels)

    @classmethod
    def from_dict(cls, obj: dict[str, Any], line: int = -1) -> LabelSample:
        labels = [BoundingBox.from_dict(label) for label in obj.pop("labels", [])]
        return cls(line=line, labels=labels, **obj)


def index_jsonl(
    jsonl: PathLike[str] | str, skip: int = 0, index: set[str] | None = None
) -> tuple[set[str], int]:
    """Returns a set of all unique ids in the given jsonl file.
    Returns both the set and the number of lines parsed."""
    if index is None:
        index = set()

    jsonl_path = Path(jsonl)
    if not jsonl_path.exists():
        return index, skip

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
    exclude: set[str] | dict[str, Any] | None = None,
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
            if obj["objectId"] in exclude:
                continue
            yield LabelSample.from_dict(obj, index)


@dataclass
class MissionResults:
    mission_dir: PathLike[str] | str = field(default_factory=Path.cwd)

    labeled_jsonl: Path = field(init=False, repr=False)
    unlabeled_jsonl: Path = field(init=False, repr=False)

    labeled: dict[str, list[BoundingBox]] = field(default_factory=dict)
    labeled_offset: int = 0

    unlabeled: list[LabelSample] = field(default_factory=list)
    unlabeled_offset: int = 0

    def __post_init__(self) -> None:
        self.labeled_jsonl = Path(self.mission_dir, "labeled.jsonl")
        self.unlabeled_jsonl = Path(self.mission_dir, "unlabeled.jsonl")

    def resync_labeled(self) -> None:
        new_labels = list(read_jsonl(self.labeled_jsonl, skip=self.labeled_offset))
        if new_labels:
            self.labeled.update((label.objectId, label.labels) for label in new_labels)
            self.labeled_offset = new_labels[-1].index

    def resync(self) -> None:
        self.resync_labeled()

        new_unlabeled = list(
            read_jsonl(self.unlabeled_jsonl, skip=self.unlabeled_offset)
        )
        if new_unlabeled:
            self.unlabeled.extend(new_unlabeled)
            self.unlabeled_offset = new_unlabeled[-1].index

    def read_unlabeled(
        self, exclude_labeled: bool = True, tail: bool = False
    ) -> Iterator[LabelSample]:
        if exclude_labeled:
            self.resync_labeled()
            exclude = self.labeled
        else:
            exclude = dict()
        yield from read_jsonl(self.unlabeled_jsonl, exclude=exclude, tail=tail)

    def save_labeled(self, results: list[LabelSample]) -> None:
        # make sure we're current with the on-disk labeled state
        self.resync_labeled()

        with self.labeled_jsonl.open("a") as fp:
            for result in results:
                # skip already labeled results
                if result.objectId in self.labeled:
                    continue

                # skip if there are still unclassified bounding boxes?
                if sum(
                    1 for bbox in result.labels if bbox.label == "" or bbox.score != 1.0
                ):
                    continue

                result.to_jsonl(fp)
