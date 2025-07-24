# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

"""Hawk Detection datatype. Defines a datatype that contains the information
about a detected object.

Provides functions to convert to/from protobuf and to/from a file.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, TypedDict

from logzero import logger

from .classes import (
    NEGATIVE_CLASS,
    ClassLabel,
    ClassList,
    ClassName,
    class_label_to_int,
    class_name_to_str,
)
from .proto import common_pb2


@dataclass
class Detection:
    class_name: ClassName
    confidence: float = 1.0
    x: float = 0.5  # center X coordinate of bounding box
    y: float = 0.5  # center Y coordinate of bounding box
    w: float = 1.0  # width of bounding box
    h: float = 1.0  # height of bounding box

    @classmethod
    def from_protobuf(cls, msg: bytes | common_pb2.Detection) -> Detection:
        """Parses a Detection from a protobuf message."""
        if isinstance(msg, bytes):
            obj = common_pb2.Detection()
            obj.ParseFromString(msg)
        else:
            obj = msg

        detection = cls(class_name=ClassName(sys.intern(obj.class_name)))

        if obj.HasField("confidence"):
            detection.confidence = obj.confidence

        if obj.HasField("coords"):
            detection.x = obj.coords.center_x
            detection.y = obj.coords.center_y
            detection.w = obj.coords.width
            detection.h = obj.coords.height

        return detection

    @classmethod
    def from_protobuf_list(cls, msg: Iterable[common_pb2.Detection]) -> list[Detection]:
        return cls.sort_detections(
            cls.from_protobuf(detection)
            for detection in msg
            if (not detection.HasField("confidence") or detection.confidence)
            and detection.class_name not in [NEGATIVE_CLASS, ""]
        )

    @classmethod
    def from_labelkit(cls, bbox: LabelKitOut, class_list: ClassList) -> Detection:
        # we have to shift the class label by one because labelkit only shows
        # positive classes as options in the pulldown.
        class_label = ClassLabel(bbox["labels"] + 1)
        min_x, min_y, max_x, max_y = bbox["bboxes"]

        return cls(
            class_name=class_list[class_label],
            x=(max_x + min_x) / 2,
            y=(max_y + min_y) / 2,
            w=(max_x - min_x),
            h=(max_y - min_y),
        )

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
            class_name=class_name,
            confidence=score,
            x=float(centerX),
            y=float(centerY),
            w=float(width),
            h=float(height),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"class_name": class_name_to_str(self.class_name)}

        if self.confidence != 1.0:
            result["confidence"] = self.confidence

        if self.w != 1.0 or self.h != 1.0:
            result.update(
                {
                    "x": self.x,
                    "y": self.y,
                    "w": self.w,
                    "h": self.h,
                },
            )

        return result

    def to_protobuf(self) -> common_pb2.Detection:
        """Returns a protobuf representation of the Detection."""
        obj = common_pb2.Detection(class_name=class_name_to_str(self.class_name))

        if self.confidence != 1.0:
            obj.confidence = self.confidence

        if self.w != 1.0 or self.h != 1.0:
            obj.coords.center_x = self.x
            obj.coords.center_y = self.y
            obj.coords.width = self.w
            obj.coords.height = self.h

        return obj

    @staticmethod
    def sort_detections(detections: Iterable[Detection]) -> list[Detection]:
        """Sort by bounding box and reversed confidence score (highest score first)."""
        return sorted(detections, key=lambda d: (d.x, d.y, d.w, d.h, -d.confidence))

    @staticmethod
    def group_detections(detections: list[Detection]) -> Iterator[Iterator[Detection]]:
        """Group by bounding box and return an iterator for each group.

        Assumes the list of detections is already sorted by boundingbox.
        """
        for _, iterator in itertools.groupby(
            detections,
            key=lambda d: (d.x, d.y, d.w, d.h),
        ):
            yield iterator

    @classmethod
    def to_labelkit(
        cls,
        detections: list[Detection],
        class_list: ClassList,
    ) -> list[LabelKitOut]:
        detections = cls.sort_detections(detections)
        return [
            next(i)._to_labelkit(class_list) for i in cls.group_detections(detections)
        ]

    def _to_labelkit(self, class_list: ClassList) -> LabelKitOut:
        class_label = class_list.index(self.class_name)
        return {
            "bboxes": (self.x, self.y, self.w, self.h),
            "labels": class_label_to_int(class_label) - 1,
        }


class LabelKitOut(TypedDict):
    bboxes: tuple[float, float, float, float]
    labels: int
