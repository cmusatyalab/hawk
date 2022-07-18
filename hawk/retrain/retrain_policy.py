# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

from abc import ABCMeta, abstractmethod


class RetrainPolicy(metaclass=ABCMeta):

    @abstractmethod
    def update(self, new_positives: int, new_negatives: int) -> None:
        self.positives += new_positives
        self.negatives += new_negatives

    @abstractmethod
    def should_retrain(self) -> bool:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

