# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Retrain policy based on absolute #labels increment
"""

from .retrain_policy_base import RetrainPolicyBase


class AbsolutePolicy(RetrainPolicyBase):

    def __init__(self, threshold: int, only_positives: bool):
        super().__init__()
        self.new_examples = 0
        self._threshold = threshold
        self._only_positives = only_positives

    def update(self, new_positives: int, new_negatives: int) -> None:
        super().update(new_positives, new_negatives)
        self.new_examples += new_positives

        if not self._only_positives:
            self.new_examples += new_negatives

    def should_retrain(self) -> bool:
        return self.new_examples >= self._threshold

    def reset(self) -> None:
        self.new_examples = 0
