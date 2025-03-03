# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Retrain policy based on availability of new model"""

import shutil
from pathlib import Path

from .retrain_policy_base import RetrainPolicyBase

FILENAME = "new_model"


class ModelPolicy(RetrainPolicyBase):
    def __init__(self, directory: str):
        super().__init__()
        self.new_examples = 0
        self._new_model_present = Path(directory, FILENAME)
        self.model_path = Path(directory, "model.pth")

    def update(self, new_positives: int, new_negatives: int) -> None:
        super().update(new_positives, new_negatives)

    def should_retrain(self) -> bool:
        return self._new_model_present.exists()

    def reset(self) -> None:
        if self.should_retrain():
            src_path = self._new_model_present.read_text().strip()
            shutil.copy(src_path, self.model_path)
            self._new_model_present.unlink()
