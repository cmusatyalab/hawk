# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from movinets import MoViNet
from movinets.config import _C
from torch import Tensor, nn

from .backbone_encoder import BackboneEncoder


class MoViNet_(MoViNet):  # type: ignore[misc]
    def __init__(
        self,
        embed_dim: int,
        cfg: Any,
        causal: bool = True,
        pretrained: bool = False,
        num_classes: int = 600,
        conv_type: str = "3d",
        tf_like: bool = False,
    ) -> None:
        super().__init__(cfg, causal, pretrained, num_classes, conv_type, tf_like)
        self.embed_dim = embed_dim
        self.classifier = nn.Linear(480, embed_dim)

    def avg(self, X: Tensor) -> Tensor:
        avg = F.adaptive_avg_pool3d(X, (X.shape[2], 1, 1))
        avg = avg.squeeze(dim=(3, 4))
        return avg

    def forward(self, X: Tensor) -> Tensor:
        B = X.shape[0]
        T = X.shape[2]
        X = self.conv1(X)
        X = self.blocks(X)
        X = self.conv7(X)
        X = self.avg(X)
        X = X.permute(0, 2, 1)  # B,T,K
        X = X.reshape(B * T, 480)
        X = self.classifier(X)
        Z = X.reshape(B, T, self.embed_dim)
        return Z


class MovinetEncoder(BackboneEncoder):

    def __init__(self, embed_dim: int):
        super().__init__(embed_dim=embed_dim)
        self._encoder = MoViNet_(
            embed_dim=embed_dim, cfg=_C.MODEL.MoViNetA0, causal=True, pretrained=True
        )

    def forward(self, X: Tensor) -> Tensor:
        """
        :param X: (B, N, 3, H, W)
        """
        B, N, C, H, W = X.shape
        assert C == 3
        self._encoder.clean_activation_buffers()
        X = X.permute(0, 2, 1, 3, 4)
        Z = self._encoder(X)  # B, N, K
        self._encoder.clean_activation_buffers()
        return Z

    def _set_embed_dim(self, dim: int) -> None:
        self._encoder.classifier = nn.Linear(480, dim)
