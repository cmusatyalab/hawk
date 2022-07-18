# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

from hawk.retrain.retrain_policy import RetrainPolicy


class AbsoluteThresholdPolicy(RetrainPolicy):

    def __init__(self, threshold: int, only_positives: bool):
        self.new_examples = 0
        self.positives = 0
        self.negatives = 0
        self._threshold = threshold
        self._only_positives = only_positives

    def update(self, new_positives: int, new_negatives: int) -> None:
        self.new_examples += new_positives

        if not self._only_positives:
            self.new_examples += new_negatives

    def should_retrain(self) -> bool:
        return self.new_examples >= self._threshold

    def reset(self) -> None:
        self.new_examples = 0
