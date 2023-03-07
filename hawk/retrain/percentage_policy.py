# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

"""Retrain policy based on percentage of #labels increment
"""

from hawk.retrain.retrain_policy_base import RetrainPolicyBase
from logzero import logger


class PercentagePolicy(RetrainPolicyBase):

    def __init__(self, threshold: float, only_positives: bool):
        super().__init__()
        self.new_examples = 0
        self._threshold = threshold
        self._only_positives = only_positives
        self._previous_size = 0

    def update(self, new_positives: int, new_negatives: int) -> None:
        super().update(new_positives, new_negatives)
        self.new_examples += new_positives

        if not self._only_positives:
            self.new_examples += new_negatives

    def should_retrain(self) -> bool:
        return self.new_examples >= (1 + self._threshold) * self._previous_size

    def reset(self) -> None:
        self._previous_size = self.new_examples
