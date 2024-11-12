# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Types and functions for managing class names and labels."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, NewType

if TYPE_CHECKING:
    # Wrapping every class name and label in a dataclass uses more memory and
    # is slower, but if we use the faster NewType, mypy still allows implicit
    # casts to the base type. So during type checking we use this version.
    @dataclass
    class ClassName:
        name: str

    @dataclass
    class ClassLabel:
        label: int

    # Used in places where we have to convert to the base type (f-strings, protobufs)
    def class_name_to_str(class_name: ClassName) -> str:
        return class_name.name

    def class_label_to_int(class_label: ClassLabel) -> int:
        return class_label.label

else:
    ClassName = NewType("ClassName", str)
    ClassLabel = NewType("ClassLabel", int)

    def class_name_to_str(class_name: ClassName) -> str:
        return class_name

    def class_label_to_int(class_label: ClassLabel) -> int:
        return class_label


# We test against this all the time
NEGATIVE_CLASS = ClassName(sys.intern("negative"))
POSITIVE_CLASS = ClassName(sys.intern("positive"))
