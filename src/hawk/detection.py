# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

"""Hawk Detection datatype. Defines a datatype that contains the information
about a detected object.

Provides functions to convert to/from protobuf and to/from a file.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from .classes import ClassName, class_name_to_str
from .proto import common_pb2


@dataclass
class Detection:
    class_name: ClassName
    confidence: float = 1.0
    center_x: float = 0.5
    center_y: float = 0.5
    width: float = 1.0
    height: float = 1.0

    @classmethod
    def from_protobuf(cls, msg: bytes | common_pb2.Detection) -> Detection:
        """Parses a Detection from a protobuf message."""
        if isinstance(msg, bytes):
            obj = common_pb2.Detection()
            obj.ParseFromString(msg)
        else:
            obj = msg

        return cls(
            class_name=ClassName(sys.intern(obj.class_name)),
            confidence=obj.confidence,
            center_x=obj.coords.center_x,
            center_y=obj.coords.center_y,
            width=obj.coords.width,
            height=obj.coords.height,
        )

    def to_protobuf(self) -> common_pb2.Detection:
        """Returns a protobuf representation of the Detection."""
        return common_pb2.Detection(
            class_name=class_name_to_str(self.class_name),
            confidence=self.confidence,
            coords=common_pb2.Region(
                center_x=self.center_x,
                center_y=self.center_y,
                width=self.width,
                height=self.height,
            ),
        )
