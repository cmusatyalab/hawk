# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

"""Retrain policy based on percentage of #tiles retrieved increment
"""

from .retrain_policy_base import RetrainPolicyBase


class SampleIntervalPolicy(RetrainPolicyBase):
    def __init__(self, num_intervals: int):
        super().__init__()
        self.new_examples = 0
        self.num_intervals = num_intervals
        self._previous_size = 0
        self.interval_samples_retrieved = 0
        self.num_retrains = num_intervals - 1
        self.total_tiles = 0
        self.retrain_num = 1

    def num_interval_sample(self, total_tiles) -> int:
        self.num_interval_samples = int(total_tiles / self.num_intervals)

    def update(self, new_positives: int, new_negatives: int) -> None:
        pass
        # super().update(new_positives, new_negatives)
        # self.new_examples += new_positives

        # if not self._only_positives:
        # self.new_examples += new_negatives

    def should_retrain(self) -> bool:
        if self.retrain_num < self.num_intervals:
            completed_interval = self.interval_samples_retrieved >= int(
                self.total_tiles / self.num_intervals
            )
            if completed_interval:
                self.retrain_num += 1
                return True
            return False
        return False

    def reset(self) -> None:
        self.interval_samples_retrieved = 0
