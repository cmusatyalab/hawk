# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import Dataset


class PTListDataset(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        list_file: str,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[int], int] | None = None,
    ) -> None:
        # read all (path, label) pairs up front
        with open(list_file) as f:
            lines = [ln.strip().split() for ln in f if ln.strip()]
        self.samples = [(p, int(lbl)) for p, lbl in lines]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        data = torch.load(path)
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label
