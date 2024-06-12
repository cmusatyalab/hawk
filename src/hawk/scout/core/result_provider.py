# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from typing import Optional

from .object_provider import ObjectProvider


class ResultProvider:
    def __init__(self, obj: ObjectProvider, score: float, model_version: Optional[int]):
        self.id = obj.id
        self.content = obj.content
        self.attributes = obj.attributes
        self.gt = obj.gt
        self.score = score
        self.model_version = model_version
