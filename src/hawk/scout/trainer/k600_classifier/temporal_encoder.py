# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn


class TransformerParams:
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        num_classes: int,
        head_dim: int,
    ):
        assert embed_dim % num_heads == 0
        assert embed_dim / num_heads == head_dim
        self.head_dim: int = head_dim
        self.embed_dim: int = embed_dim
        self.depth: int = depth
        self.num_heads: int = num_heads
        self.mlp_dim: int = mlp_dim
        self.num_classes: int = num_classes

    def __str__(self) -> str:
        return f"TransformerParams = {self.__dict__}"


def posemb_sincos_1d(tokens: torch.Tensor, temperature: int = 10000) -> torch.Tensor:
    _, N, dim = tokens.shape
    device, dtype = tokens.device, tokens.dtype

    N = torch.arange(N, device=device)
    assert (dim % 2) == 0, "feature dimension must be multiple of 2 for sincos emb"
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    N = N.flatten()[:, None] * omega[None, :]
    pe = torch.cat((N.sin(), N.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class SimpleViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int,
    ):
        super().__init__()

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,num_frames, dim)
        x = x + posemb_sincos_1d(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.linear_head(x)
