# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Types and functions for managing class names and labels."""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NewType

from logzero import logger

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


@dataclass
class ClassCounter:
    class_list: list[ClassName]
    counter: Counter[ClassName] = field(default_factory=Counter)

    def count(self, label: ClassLabel, count: int = 1) -> None:
        try:
            class_name = self.class_list[class_label_to_int(label)]
            self.counter[class_name] += count
        except IndexError:
            logger.error("Unknown class {label} encountered")

    def update(self, counts: dict[ClassName, int]) -> None:
        self.counter.update(counts)

    @property
    def negatives(self) -> int:
        return self.counter[NEGATIVE_CLASS]

    @property
    def positives(self) -> int:
        return sum(self.counter.values()) - self.negatives

    def __repr__(self) -> str:
        return str(dict(self.counter))
