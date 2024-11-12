# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from ...classes import ClassLabel, ClassName


class MLClass:
    def __init__(self, name: ClassName, label: ClassLabel):
        self.name = name
        self.label = label
        self.total_samples = 0

    def add_samples(self, count: int) -> None:
        if count < 0:
            raise ValueError("Count of samples cannot be negative.")
        self.total_samples += count

    def __repr__(self) -> str:
        return (
            f"MLClass(name={self.name}, label={self.label}, "
            f"total_samples={self.total_samples})"
        )


class MLClassManager:
    def __init__(self) -> None:
        self.classes: dict[ClassName, MLClass] = {}
        self.class_list: list[MLClass] = []
        self.label_name_dict: dict[ClassLabel, ClassName] = {}

    def add_class(self, name: ClassName, label: ClassLabel) -> None:
        if name in self.classes:
            msg = f"Class {name} already exists."
            raise ValueError(msg)
        if label in self.label_name_dict:
            msg = f"Label {label} already exists."
            raise ValueError(msg)
        new_class = MLClass(name, label)
        self.classes[name] = new_class
        self.class_list.append(new_class)
        self.label_name_dict[label] = name

    def get_class(self, name: ClassName) -> MLClass | None:
        return self.classes.get(name)

    def get_labels(self) -> list[str]:
        return [str(cls.label) for cls in self.class_list]

    def add_samples(self, name: ClassName, count: int) -> None:
        ml_class = self.get_class(name)
        if ml_class is None:
            msg = f"Class {name} does not exist."
            raise ValueError(msg)
        ml_class.add_samples(count)

    def get_total_samples(self) -> int:
        return sum(ml_class.total_samples for ml_class in self.classes.values())

    def get_total_positives(self) -> int:
        return sum(
            ml_class.total_samples
            for ml_class in self.classes.values()
            if ml_class.label != ClassLabel(0)
        )

    def __repr__(self) -> str:
        return f"MLClassManager(classes={list(self.classes.values())})"


"""
# Example usage
>>> manager = MLClassManager()
>>> manager.add_class("Negatives", 0)
>>> manager.add_class("Roundabout", 1)
>>> manager.add_class("Pool", 2)

# Add samples to the classes via manager
>>> manager.add_samples("Negatives", 100)
>>> manager.add_samples("Roundabout", 150)
>>> manager.add_samples("Pool", 200)

>>> print(manager)
MLClassManager(classes=[
   MLClass(name=Class1, label=0, total_samples=100),
   MLClass(name=Class2, label=1, total_samples=150),
   MLClass(name=Class3, label=2, total_samples=200)])

# Update samples
>>> manager.add_samples("Negatives", 50)
>>> print(manager.get_class("Negatives"))
MLClass(name=Class1, label=0, total_samples=150)

# Get total samples across all classes
>>> total_samples = manager.get_total_samples()
>>> print(f"Total samples across all classes: {total_samples}")
Total samples across all classes: 500

>>> total_positive_samples = manager.get_total_positives()
>>> print(f"Total positive samples: {total_positive_samples}")
Total positives samples across all classes: 500
"""
