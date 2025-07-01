# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from torch import Tensor, nn


class BackboneEncoder(nn.Module):  # type: ignore[misc]

    def __init__(self, embed_dim: int):
        super().__init__()
        self._embed_dim: int = embed_dim

    def forward(self, X: Tensor) -> Tensor:
        """
        :param X: (B, N, C, H, W)
        """
        pass

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @embed_dim.setter
    def embed_dim(self, embed_dim: int) -> None:
        self._set_embed_dim(embed_dim)
        self._embed_dim = embed_dim

    def _set_embed_dim(self, embed_dim: int) -> None:
        pass
