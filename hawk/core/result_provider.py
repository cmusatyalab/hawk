# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

from typing import Optional

from hawk.core.object_provider import ObjectProvider

class ResultProvider(object):

    def __init__(self, obj_id: str, label: str, score: float, model_version: Optional[int], 
                 obj: ObjectProvider):
        self.id = obj_id
        self.label = label
        self.score = score
        self.content = obj.content
        self.attributes = obj.attributes
        self.gt = obj.gt
        self.model_version = model_version
