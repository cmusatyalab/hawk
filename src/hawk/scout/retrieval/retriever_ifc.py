# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterator


class RetrieverIfc:

    @abstractmethod
    def get_ml_ready_data(
        self, object_ids: list[str | int] | str | int
    ) -> list[tuple[Any, str | int]] | tuple[Any, str | int]: ...

    @abstractmethod
    def get_oracle_ready_data(
        self, object_ids: list[str | int] | str | int
    ) -> list[tuple[Any, str | int]] | tuple[Any, str | int]: ...

    @abstractmethod
    def object_ids_stream(self) -> Iterator[str | int]: ...  ## Generator

    @abstractmethod
    def get_ground_truth(
        self, object_ids: list[str | int] | str | int
    ) -> list[tuple[int, str | int]] | tuple[int, str | int]: ...
