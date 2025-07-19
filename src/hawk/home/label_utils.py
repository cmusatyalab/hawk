# SPDX-FileCopyrightText: 2022-2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Utility functions for labelers"""

from __future__ import annotations

import dataclasses
import json
import time
from collections import Counter
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, TextIO, TypedDict

from logzero import logger

from ..classes import (
    NEGATIVE_CLASS,
    ClassList,
    ClassName,
    class_name_to_str,
)
from ..detection import Detection
from ..hawkobject import MEDIA_TYPES
from ..objectid import ObjectId
from ..rusty import map_
from .utils import tailf

if TYPE_CHECKING:
    from collections.abc import Container
    from os import PathLike


# seems to be a (non-dataclass) superset of hawk.detection.Detection
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
        detections = [Detection(**d) for d in obj.pop("detections", [])]
        groundtruth = [Detection(**gt) for gt in obj.pop("groundtruth", [])]
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

        if class_list is not None:
            class_list.extend(d.class_name for d in self.detections)
            class_list.extend(gt.class_name for gt in self.groundtruth)

        jsonl = json.dumps(
            dict(
                scoutIndex=self.scoutIndex,
                model_version=self.model_version,
                queued=self.queued,
                oracle_items=self.oracle_items,
                detections=[d.to_dict() for d in self.detections],
                groundtruth=[gt.to_dict() for gt in self.groundtruth],
                **kwargs,
            )
        )
        fp.write(f"{jsonl}\n")

    def to_labelkit_args(self, class_list: ClassList) -> LabelKitArgs:
        bboxes = Detection.to_labelkit(self.detections, class_list)
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
            result["class_name"] = class_name_to_str(detection.class_name)
            result["confidence"] = detection.confidence
            result["bbox_x"] = detection.x
            result["bbox_y"] = detection.y
            result["bbox_w"] = detection.w
            result["bbox_h"] = detection.h
            # because we're modifying result in-place as we iterate
            # we need to return a copy here
            yield result.copy()

    @property
    def classes(self) -> set[ClassName]:
        return set(detection.class_name for detection in self.detections)

    def class_counts(self) -> Counter[ClassName]:
        count: Counter[ClassName] = Counter(
            detection.class_name for detection in self.detections
        )
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
            max(detection.confidence for detection in self.detections)
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
                skip = False
                for detection in result.detections:
                    if detection.confidence != 1.0:
                        skip = True
                if skip:
                    continue

                result.to_jsonl(fp)
