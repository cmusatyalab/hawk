# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Types and functions for managing class names and labels."""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import InitVar, dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Iterable, Iterator, NewType, Sequence

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
class ClassList:
    """List of (known unique) class names, "negative" is always class 0"""

    classes: InitVar[Iterable[str]] = []

    def __post_init__(self, classes: Iterable[str]) -> None:
        # Ensure we always have 'negative' as class 0
        self.lock = Lock()
        self._classes = [NEGATIVE_CLASS]
        self.extend(ClassName(sys.intern(name)) for name in classes)

    def __repr__(self) -> str:
        return repr(self._classes)

    def __getitem__(self, class_label: ClassLabel) -> ClassName:
        """class-label to class-name lookup.
        Raises IndexError when class_label is not found.
        """
        return self._classes[class_label_to_int(class_label)]

    def index(self, class_name: ClassName) -> ClassLabel:
        """class-name to class-label lookup.
        Raises ValueError if class_name is not found.
        """
        return ClassLabel(self._classes.index(class_name))

    @property
    def positive(self) -> Sequence[ClassName]:
        """Returns a list of only positive class names."""
        return self._classes[1:]

    def __contains__(self, class_name: ClassName) -> bool:
        return class_name in self._classes

    def __iter__(self) -> Iterator[ClassName]:
        yield from self._classes

    def __len__(self) -> int:
        return len(self._classes)

    def add(self, class_name: ClassName) -> None:
        """Add a single new class if it doesn't exist already."""
        if class_name not in self._classes:
            with self.lock:
                if class_name not in self._classes:
                    self._classes.append(class_name)

    def extend(self, classes: Iterable[ClassName]) -> ClassList:
        """Extend with new class names."""
        # append one at a time in case there are duplicates
        for class_name in classes:
            self.add(class_name)
        return self


@dataclass
class ClassCounter:
    class_list: ClassList
    counter: Counter[ClassName] = field(default_factory=Counter)

    def count(self, label: ClassLabel, count: int = 1) -> None:
        try:
            class_name = self.class_list[label]
            self.counter[class_name] += count
        except IndexError:
            logger.error("Unknown class {label} encountered")

    def update(
        self, counts: Iterable[ClassName] | dict[ClassName, int] | ClassCounter
    ) -> None:
        if isinstance(counts, ClassCounter):
            counts = counts.counter
        self.counter.update(counts)

    def total_sum(self) -> int:
        return sum(self.counter.values())

    @property
    def negatives(self) -> int:
        return self.counter[NEGATIVE_CLASS]

    @property
    def positives(self) -> int:
        return self.total_sum() - self.negatives

    def __repr__(self) -> str:
        return str(dict(self.counter))

    def __bool__(self) -> bool:
        return bool(self.counter)

    def copy(self) -> ClassCounter:
        return ClassCounter(class_list=self.class_list, counter=self.counter.copy())
