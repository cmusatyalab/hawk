# SPDX-FileCopyrightText: 2022-2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Utility functions for labelers"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, NewType, TextIO

from logzero import logger

from .utils import tailf

if TYPE_CHECKING:
    from collections.abc import Container
    from os import PathLike


ClassName = NewType("ClassName", str)


@dataclass
class ClassMap:
    """ClassMap contains a list of class names, it tracks positive classes only."""

    classes: list[ClassName]

    @classmethod
    def from_list(cls, classes: list[str]) -> ClassMap:
        """Create a list of all positive class names"""
        return cls(
            classes=[
                ClassName(sys.intern(name))
                for name in classes
                if name not in ["", "neg", "negative"]
            ]
        )

    def __getitem__(self, item: int | str) -> ClassName:
        """Tries to accept either numeric labels or names and maps them to class names
        Will create new class labels for unknown class names.
        Raises IndexError when we see a class label that we have no name for.
        """
        try:
            class_label = int(item)
        except ValueError:
            class_label = self.name_to_label(str(item))
        return self.classes[class_label]

    def name_to_label(self, name: str) -> int:
        """Returns label for the given class.
        Creates a new label for the class if it did not previously exist.
        """
        class_name = ClassName(name)
        try:
            return self.classes.index(class_name)
        except ValueError:
            new_label = len(self.classes)
            logger.warning(f"Adding unknown class {class_name} with label {new_label}")
            self.classes.append(class_name)
            return new_label


@dataclass(order=True)
class Detection:
    # normalized center x/y/width/height output as used in yolo
    x: float = 0.5
    y: float = 0.5
    w: float = 1.0
    h: float = 1.0

    # we only compare regions, so two detections with different scores compare as equal!
    cls_scores: dict[ClassName, float] = field(default_factory=dict, compare=False)

    minX: InitVar[float | None] = None
    minY: InitVar[float | None] = None
    maxX: InitVar[float | None] = None
    maxY: InitVar[float | None] = None

    def __post_init__(
        self,
        minX: float | None,
        minY: float | None,
        maxX: float | None,
        maxY: float | None,
    ) -> None:
        if minX is not None and maxX is not None:
            self.x = (maxX + minX) / 2
            self.w = maxX - minX
        if minY is not None and maxY is not None:
            self.y = (maxY + minY) / 2
            self.h = maxY - minY

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> Detection:
        # filter out negatives (and 0 scores)
        cls_scores = {
            ClassName(sys.intern(cls)): score
            for cls, score in obj.pop("cls_scores", {}).items()
            if score and cls not in ["", "neg", "negative"]
        }
        return cls(cls_scores=cls_scores, **obj)

    @classmethod
    def from_yolo(cls, line: str, class_map: ClassMap) -> Detection:
        label, centerX, centerY, width, height, *_score = line.split()

        class_name = class_map[label]
        score = float(_score[0]) if _score else 1.0

        return cls(
            x=float(centerX),
            y=float(centerY),
            w=float(width),
            h=float(height),
            cls_scores={class_name: score},
        )

    def to_dict(self, class_map: ClassMap | None = None) -> dict[str, Any]:
        """asdict but if class_map is given we add missing classes to cls_scores."""
        # don't include negative class from class_map
        classes: Iterable[ClassName] = (
            class_map.classes if class_map is not None else self.classes
        )
        return dict(
            x=self.x,
            y=self.y,
            w=self.w,
            h=self.h,
            cls_scores={cls: self.cls_scores.get(cls, 0.0) for cls in classes},
        )

    def to_yoloish(self, class_map: ClassMap) -> Iterator[str]:
        """Yolo (and ClassMap) index positive classes counting from 0.
        But scouts are 1-indexed because they include an implicit negative class 0.
        """
        for class_name in self.cls_scores:
            label = class_map.name_to_label(class_name) + 1
            # TODO send class_name instead of class_label to the scout
            yield f"{label} {self.x} {self.y} {self.w} {self.h}"

    @property
    def classes(self) -> set[ClassName]:
        return {cls for cls in self.cls_scores}

    @property
    def has_scores(self) -> bool:
        return sum(self.cls_scores.values()) != 0.0

    @property
    def max_score(self) -> float:
        return max(self.cls_scores.values())

    @classmethod
    def merge_detections(cls, detections: Iterator[Detection]) -> Iterator[Detection]:
        prev: Detection | None = None

        for cur in sorted(detections):
            # we only compare the regions (xywh) and not the scores
            if prev == cur:
                assert prev is not None
                prev.cls_scores.update(cur.cls_scores)
                continue

            # drop detections with no scores?
            if prev is not None and prev.has_scores:
                yield prev
            prev = cur

        # and yield any remaining detection
        if prev is not None and prev.has_scores:
            yield prev


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

    @classmethod
    def from_dict(cls, obj: dict[str, Any], line: int = -1) -> LabelSample:
        detections = [
            Detection.from_dict(detection) for detection in obj.pop("detections", [])
        ]
        return cls(line=line, detections=detections, **obj)

    def to_jsonl(self, fp: TextIO, class_map: ClassMap | None = None) -> None:
        jsonl = json.dumps(
            dict(
                objectId=self.objectId,
                scoutIndex=self.scoutIndex,
                queued=self.queued,
                detections=[
                    detection.to_dict(class_map) for detection in self.detections
                ],
            )
        )
        fp.write(f"{jsonl}\n")

    def to_yoloish(self, class_map: ClassMap) -> str:
        """Yolo (and class_map) use class labels that start counting at 0
        But hawk scouts count positive classes from 1, so this is 'yolo-ish'
        """
        return "".join(
            line + "\n"
            for detection in self.detections
            for line in detection.to_yoloish(class_map)
        )

    @property
    def classes(self) -> set[ClassName]:
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
    def classes(self) -> set[ClassName]:
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
