# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

from hawk.retrain.retrain_policy import RetrainPolicy
from logzero import logger


class PercentageThresholdPolicy(RetrainPolicy):

    def __init__(self, threshold: float, only_positives: bool):
        self.new_examples = 0
        self.positives = 0
        self.negatives = 0
        self._threshold = threshold
        self._only_positives = only_positives
        self._previous_size = 0

    def update(self, new_positives: int, new_negatives: int) -> None:
        self.new_examples += new_positives
        self.positives += new_positives
        self.negatives += new_negatives

        if not self._only_positives:
            self.new_examples += new_negatives

    def should_retrain(self) -> bool:
        logger.info("Examples {} >= (1 + threshold {}) * self._previous_size {} = {}".format(
            self.new_examples, self._threshold, self._previous_size,
            (self.new_examples >= (1 + self._threshold) * self._previous_size)))
        return self.new_examples >= (1 + self._threshold) * self._previous_size

    def reset(self) -> None:
        self._previous_size = self.new_examples
