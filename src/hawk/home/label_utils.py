# SPDX-FileCopyrightText: 2022-2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Utility functions for labelers"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from collections import Counter
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, TextIO, TypedDict

from logzero import logger

from ..classes import (
    NEGATIVE_CLASS,
    ClassLabel,
    ClassList,
    ClassName,
    class_label_to_int,
    class_name_to_str,
)
from ..hawkobject import MEDIA_TYPES
from ..objectid import ObjectId
from ..rusty import map_
from .utils import tailf

if TYPE_CHECKING:
    from collections.abc import Container
    from os import PathLike


class DetectionDict(TypedDict):
    time_queued: float  # Unix time in seconds
    instance: int  # unique counter, combine w. object_id to spot relabel events
    object_id: str | None  # unique object identifier
    scout_index: int  # index of scout where sample originated
    model_version: int  # model version used to inference
    image_path: str  # path to image
    class_name: str  # class label
    confidence: float  # confidence of inference (1.0 for labeled results)
    bbox_x: float  # center x-coordinate of bounding box area (0.5 for classification)
    bbox_y: float  # center y-coordinate of bounding box area (0.5 for classification)
    bbox_w: float  # width of bounding box area (1.0 for classification)
    bbox_h: float  # height of bounding box area (1.0 for classification)


class LabelKitArgs(TypedDict):
    bboxes: list[tuple[float, float, float, float]]
    labels: list[int]


class LabelKitOut(TypedDict):
    bboxes: tuple[float, float, float, float]
    labels: int


@dataclass(order=True)
class Detection:
    # normalized center x/y/width/height output as used in yolo
    x: float = 0.5
    y: float = 0.5
    w: float = 1.0
    h: float = 1.0

    # we only compare regions, so two detections with different scores compare as equal!
    scores: dict[ClassName, float] = field(default_factory=dict, compare=False)

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
    def from_boundingbox(
        cls, x: float, y: float, w: float, h: float, class_name: str, confidence: float
    ) -> Detection:
        _class_name = ClassName(sys.intern(class_name))
        return cls(x=x, y=y, w=w, h=h, scores={_class_name: confidence})

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> Detection:
        # filter out negatives (and 0 scores)
        scores = {
            ClassName(sys.intern(name)): score
            for name, score in obj.pop("scores", obj.pop("cls_scores", {})).items()
            if score and name not in ["", "neg", "negative"]
        }
        return cls(scores=scores, **obj)

    @classmethod
    def from_yolo(cls, line: str, class_list: ClassList) -> Detection:
        label, centerX, centerY, width, height, *_score = line.split()

        try:
            class_label = ClassLabel(int(label))
            class_name = class_list[class_label]
        except ValueError:
            class_name = ClassName(label)
            class_list.add(class_name)
        except IndexError:
            # Label was numeric, but we couldn't find it in the class list.
            # All classes should be known and named at this point.
            logger.error("Unexpected class label {label} encountered")
            raise

        score = float(_score[0]) if _score else 1.0

        return cls(
            x=float(centerX),
            y=float(centerY),
            w=float(width),
            h=float(height),
            scores={class_name: score},
        )

    @classmethod
    def from_labelkit(cls, obj: LabelKitOut, class_list: ClassList) -> Detection:
        # we have to shift the class label by one because labelkit only shows
        # positive classes as options in the pulldown.
        class_label = ClassLabel(obj["labels"] + 1)
        class_name = class_list[class_label]
        return cls(*obj["bboxes"], scores={class_name: 1.0})

    def to_labelkit(self, class_list: ClassList) -> LabelKitOut:
        class_label = class_list.index(self.top_class())
        return dict(
            bboxes=(self.x, self.y, self.w, self.h),
            labels=class_label_to_int(class_label) - 1,
        )

    def to_dict(self, class_list: ClassList | None = None) -> dict[str, Any]:
        """asdict but if list of classes is given we add all classes to scores."""
        # don't include negative class from class_map
        # add any new classes to the class list
        if class_list is not None:
            class_list.extend(self.scores)
            positives = class_list.positive
        else:
            positives = list(self.scores)

        return dict(
            x=self.x,
            y=self.y,
            w=self.w,
            h=self.h,
            scores={cls: self.scores.get(cls, 0.0) for cls in positives},
        )

    def by_score(self) -> list[tuple[ClassName, float]]:
        return sorted(self.scores.items(), key=lambda x: x[1], reverse=True)

    def class_counts(self) -> Counter[ClassName]:
        return Counter(self.scores)

    def class_list(self) -> ClassList:
        return ClassList().extend(self.scores)

    @property
    def has_scores(self) -> bool:
        return sum(self.scores.values()) != 0.0

    def top_class(self) -> ClassName:
        return self.by_score()[0][0]

    @property
    def max_score(self) -> float:
        return max(self.scores.values())

    @property
    def min_X(self) -> float:
        return self.x - self.w / 2

    @property
    def min_Y(self) -> float:
        return self.y - self.h / 2

    @classmethod
    def merge_detections(cls, detections: Iterator[Detection]) -> Iterator[Detection]:
        prev: Detection | None = None

        for cur in sorted(detections):
            # we only compare the regions (xywh) and not the scores
            if prev == cur:
                assert prev is not None
                prev.scores.update(cur.scores)
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

    objectId: ObjectId | None  # unique object id (None if this is a new example)
    scoutIndex: int  # index of originating scout
    model_version: int = -1  # version of the model used to generate the sample
    queued: float = field(default_factory=time.time)
    oracle_items: list[str] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    groundtruth: list[Detection] = field(default_factory=list)
    novel_sample: bool = False
    line: InitVar[int] = -1  # used to track line number in jsonl file
    image_name: InitVar[Path | None] = None  # must be defined for new examples

    def __post_init__(self, line: int, image_name: Path | None) -> None:
        if self.objectId is not None:
            image_name = self.objectId.file_name()
        else:
            assert image_name is not None
        self._image_name = image_name
        self.index = line

        # backward compatibility
        if not self.oracle_items:
            self.oracle_items = ["image/jpeg"]

    @classmethod
    def from_dict(cls, obj: dict[str, Any], line: int = -1) -> LabelSample:
        if "objectId" in obj:
            object_id = obj["objectId"]
            if isinstance(object_id, str):
                obj["objectId"] = ObjectId(object_id)
        detections = [
            Detection.from_dict(detection) for detection in obj.pop("detections", [])
        ]
        groundtruth = [Detection.from_dict(gt) for gt in obj.pop("groundtruth", [])]
        return cls(line=line, detections=detections, groundtruth=groundtruth, **obj)

    def replace(self, detections: list[Detection]) -> LabelSample:
        return dataclasses.replace(self, detections=detections, line=self.index)

    def to_jsonl(
        self,
        fp: TextIO,
        class_list: ClassList | None = None,
        **kwargs: int | str | float,
    ) -> None:
        if self.objectId is not None:
            kwargs["objectId"] = self.objectId.serialize_oid()
        else:
            kwargs["image_name"] = str(self._image_name)

        jsonl = json.dumps(
            dict(
                scoutIndex=self.scoutIndex,
                model_version=self.model_version,
                queued=self.queued,
                oracle_items=self.oracle_items,
                detections=[
                    detection.to_dict(class_list) for detection in self.detections
                ],
                groundtruth=[
                    groundtruth.to_dict(class_list) for groundtruth in self.groundtruth
                ],
                **kwargs,
            )
        )
        fp.write(f"{jsonl}\n")

    def to_labelkit_args(self, class_list: ClassList) -> LabelKitArgs:
        bboxes = [bbox.to_labelkit(class_list) for bbox in self.detections]
        return dict(
            bboxes=[out["bboxes"] for out in bboxes],
            labels=[out["labels"] for out in bboxes],
        )

    def to_flat_dict(
        self, index: int = 0, image_dir: Path | None = None
    ) -> Iterator[DetectionDict]:
        """Yields a list of dicts where each dict contains a single
        object/boundingbox/class/confidence detection event.
        Mostly useful when building a Pandas dataframe.
        """
        if image_dir is None:
            image_dir = Path(".")

        result: DetectionDict = dict(
            time_queued=self.queued,
            instance=index,
            object_id=map_(self.objectId, lambda o: o.serialize_oid()),
            scout_index=self.scoutIndex,
            model_version=self.model_version,
            image_path=str(self.content(image_dir, index=0)),
            class_name=class_name_to_str(NEGATIVE_CLASS),
            confidence=1.0,
            bbox_x=0.5,
            bbox_y=0.5,
            bbox_w=1.0,
            bbox_h=1.0,
        )
        if not self.detections:
            yield result
            return
        for detection in self.detections:
            result["bbox_x"] = detection.x
            result["bbox_y"] = detection.y
            result["bbox_w"] = detection.w
            result["bbox_h"] = detection.h
            for cls, confidence in detection.scores.items():
                result["class_name"] = str(cls)
                result["confidence"] = confidence
                # because we're modifying result in-place as we iterate the
                # bounding boxes and inference results we need to copy here
                yield result.copy()

    @property
    def classes(self) -> set[ClassName]:
        return set.union(
            set(), *[detection.class_list() for detection in self.detections]
        )

    def class_counts(self) -> Counter[ClassName]:
        count: Counter[ClassName] = Counter()
        for detection in self.detections:
            count.update(detection.class_counts())
        if not count:
            count[NEGATIVE_CLASS] = 1
        return count

    @property
    def is_classification(self) -> bool:
        # heuristically a single detection covering the whole image should be a
        # classification result
        return (
            len(self.detections) == 1
            and self.detections[0].w == 1.0
            and self.detections[0].h == 1.0
        )

    @property
    def max_score(self) -> float:
        return (
            max(detection.max_score for detection in self.detections)
            if self.detections
            else 0.0
        )

    def content(
        self, directory: Path, index: int | None = None, suffix: str | None = None
    ) -> Path:
        path = directory / self._image_name
        if index is not None:
            media_type = self.oracle_items[index]
            suffix = MEDIA_TYPES[media_type][0]

        return path.with_suffix(suffix) if suffix is not None else path


def index_jsonl(
    jsonl: PathLike[str] | str, skip: int = 0, index: set[ObjectId] | None = None
) -> tuple[set[ObjectId], int]:
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
            index.add(ObjectId(obj["objectId"]))
    return index, n


def read_jsonl(
    jsonl: PathLike[str] | str,
    exclude: Container[ObjectId] | None = None,
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
            objectId = ObjectId(obj["objectId"])
            if objectId in exclude:
                logger.debug(f"Not loading previously labeled {objectId}")
                continue
            yield LabelSample.from_dict(obj, index)


@dataclass
class MissionData:
    mission_dir: Path = field(default_factory=Path.cwd)

    labeled_jsonl: Path = field(init=False, repr=False)
    unlabeled_jsonl: Path = field(init=False, repr=False)

    labeled: dict[ObjectId, LabelSample] = field(default_factory=dict)
    labeled_offset: int = 0

    unlabeled: list[LabelSample] = field(default_factory=list)
    unlabeled_offset: int = 0

    def __post_init__(self) -> None:
        self.labeled_jsonl = self.mission_dir / "labeled.jsonl"
        self.unlabeled_jsonl = self.mission_dir / "unlabeled.jsonl"

    def resync_labeled(self) -> None:
        new_labels = list(read_jsonl(self.labeled_jsonl, skip=self.labeled_offset))
        if new_labels:
            self.labeled.update(
                (label.objectId, label)
                for label in new_labels
                if label.objectId is not None
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
        scout_classes = self.unlabeled[-1].classes if self.unlabeled else set()
        local_classes = {
            cls for result in self.labeled.values() for cls in result.classes
        }
        return scout_classes | local_classes

    def read_unlabeled(
        self, exclude_labeled: bool = True, tail: bool = False
    ) -> Iterator[LabelSample]:
        if exclude_labeled:
            self.resync_labeled()
            exclude: Container[ObjectId] = self.labeled
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
                    for cls, score in detection.scores.items():
                        if not cls or score != 1.0:
                            continue

                result.to_jsonl(fp)
