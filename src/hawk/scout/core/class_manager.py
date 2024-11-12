# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from ...classes import ClassLabel, ClassName


class MLClass:
    def __init__(self, name: ClassName, label: ClassLabel):
        self.name = name
        self.label = label

    def __repr__(self) -> str:
        return f"MLClass(name={self.name}, label={self.label}"


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
"""
