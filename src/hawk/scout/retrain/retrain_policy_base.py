# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Abstract class for retrain policies
"""

from abc import ABCMeta, abstractmethod


class RetrainPolicyBase(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.positives = 0
        self.negatives = 0

    @abstractmethod
    def update(self, new_positives: int, new_negatives: int) -> None:
        """Updating the number of labeled items"""
        self.positives += new_positives
        self.negatives += new_negatives

    @abstractmethod
    def should_retrain(self) -> bool:
        """Checks if model retain condition satisfied"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the retrain policy"""
        pass
