# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

from typing import Optional

from .attribute_provider import AttributeProvider


class ObjectProvider(object):

    def __init__(self, obj_id: str, content: bytes, 
                 attributes: AttributeProvider, gt: int=0):
        self.id = obj_id
        self.content = content
        self.attributes = attributes
        self.gt = gt
