# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterator

from ...objectid import ObjectId
from ..core.result_provider import BoundingBox


class RetrieverIfc:

    @abstractmethod
    def object_ids_stream(self) -> Iterator[ObjectId]: ...

    @abstractmethod
    def get_ml_ready_data(self, object_id: ObjectId) -> tuple[Any, str]: ...

    @abstractmethod
    def get_oracle_ready_data(self, object_id: ObjectId) -> list[tuple[Any, str]]: ...

    @abstractmethod
    def get_ground_truth(self, object_id: ObjectId) -> list[BoundingBox]: ...
