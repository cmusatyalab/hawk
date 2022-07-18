# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

from hawk.retrain.retrain_policy import RetrainPolicy
import shutil
import os

FILENAME = "new_model"


class ModelPolicy(RetrainPolicy):

    def __init__(self, directory: str):
        self.new_examples = 0
        self.positives = 0
        self.negatives = 0
        self._new_model_present = os.path.join(directory, FILENAME)
        self.model_path = os.path.join(directory, "model.pth")

    def update(self, new_positives: int, new_negatives: int) -> None:
        super().update(new_positives, new_negatives)

    def should_retrain(self) -> bool:
        return os.path.exists(self._new_model_present)

    def reset(self) -> None:
        if os.path.exists(self._new_model_present):
            src_path = open(self._new_model_present).read().strip()
            shutil.copy(src_path, self.model_path)
            os.remove(self._new_model_present)

