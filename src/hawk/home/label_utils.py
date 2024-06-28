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

from .utils import tailf

if TYPE_CHECKING:
    from os import PathLike


@dataclass
class BoundingBox:
    label: int
    score: float = 1.0
    minX: float = 0.0
    minY: float = 0.0
    maxX: float = 1.0
    maxY: float = 1.0

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> BoundingBox:
        return cls(**obj)

    @classmethod
    def from_yolo(cls, line: str) -> BoundingBox:
        label, centerX_, centerY_, width, height, *score = line.split()
        centerX, centerY = float(centerX_), float(centerY_)
        deltaX, deltaY = float(width) / 2, float(height) / 2
        return cls(
            label=int(label) + 1,
            minX=centerX - deltaX,
            minY=centerY - deltaY,
            maxX=centerX + deltaX,
            maxY=centerY + deltaY,
            score=float(score[0]) if score else 1.0,
        )

    def to_yolo(self) -> str:
        centerX = (self.maxX + self.minX) / 2
        centerY = (self.maxY + self.minY) / 2
        deltaX = self.maxX - self.minX
        deltaY = self.maxY - self.minY
        return f"{self.label} {centerX} {centerY} {deltaX} {deltaY} {self.score}"


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

    def to_yolo(self) -> str:
        return "".join(label.to_yolo() + "\n" for label in self.labels)

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
                    1 for bbox in result.labels if bbox.label < 0 or bbox.score != 1.0
                ):
                    continue

                result.to_jsonl(fp)
