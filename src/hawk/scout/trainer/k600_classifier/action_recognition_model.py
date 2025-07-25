# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import torch
from torch import Tensor, nn

from .backbone_encoder import BackboneEncoder
from .temporal_encoder import SimpleViT, TransformerParams

if TYPE_CHECKING:
    import os

Encoder = TypeVar("Encoder", bound=BackboneEncoder)


class ActionRecognitionModel(nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        encoder: Encoder,
        transformer_params: TransformerParams,
        T: int,
        stride: int,
    ) -> None:
        super().__init__()
        assert transformer_params.embed_dim == encoder.embed_dim
        embed_dim = transformer_params.embed_dim
        self.stride: int = stride
        self.T: int = T
        self.embed_dim: int = embed_dim
        self._encoder: BackboneEncoder = encoder
        self._transformer_params: TransformerParams = transformer_params
        self.num_classes: int = transformer_params.num_classes
        head_dim = transformer_params.head_dim
        self._ln = nn.LayerNorm(normalized_shape=self.embed_dim)
        self._temporal_enc = SimpleViT(
            dim=embed_dim,
            depth=transformer_params.depth,
            num_classes=transformer_params.num_classes,
            mlp_dim=transformer_params.mlp_dim,
            heads=transformer_params.num_heads,
            dim_head=head_dim,
        )

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """:param X: (B,N,C,H,W) tensor
        :return: (B,num_classes) logits tensor, where B is the batch size,
                 and (B,n_clips,T,K) Z-embedding tensor, where n_clips=(N-T)//stride +1
        """
        B, N, C, H, W = X.shape
        Z = self._encoder(X)
        assert Z.shape == (B, N, self.embed_dim)
        Z_ = self._ln(Z)
        logits = self._temporal_enc(Z_)
        Z = Z.unfold(dimension=1, size=self.T, step=self.stride).permute(0, 1, 3, 2)
        n_clips = (N - self.T) // self.stride + 1
        assert Z.shape == (B, n_clips, self.T, self.embed_dim)
        return logits, Z

    def save(self, model_path: str, num_samples: int) -> None:
        snapshot = {
            "model_state": self.state_dict(),
            "num_samples": num_samples,
            "transformer_params": self._transformer_params,
            "backbone": self._encoder.__class__,
            "T": self.T,
            "stride": self.stride,
            "version": 7,
        }
        torch.save(snapshot, model_path)
        print(f"Saved snapshot to {model_path}, num_samples = {num_samples}")

    @staticmethod
    def load(
        model_path: Path,
        backbone_encoder: BackboneEncoder,
        patch: Any | None = None,
    ) -> tuple[ActionRecognitionModel, int]:
        if model_path.suffix in [".pth", ".pt"]:
            snapshot: dict[str, Any] = torch.load(
                str(model_path),
                map_location="cpu",
                pickle_module=patch,
            )
            transformer_params = snapshot["transformer_params"]
            model: ActionRecognitionModel = ActionRecognitionModel(
                backbone_encoder,
                transformer_params,
                snapshot["T"],
                snapshot["stride"],
            )
            if snapshot["version"] == 6:
                print("Loading v6 model")
                model._temporal_enc.norm = nn.LayerNorm(transformer_params.embed_dim)
            model.load_state_dict(snapshot["model_state"])
            if snapshot["version"] == 6:
                print("Removing unused Layer Norm")
                del model._temporal_enc.norm
            num_samples = snapshot["num_samples"]
        else:
            msg = f"Illegal extension {model_path.suffix}"
            raise ValueError(msg)
        print(
            f"Loaded action recognition model {model_path}"
            f" with num_samples={num_samples}",
        )
        return model, num_samples

    @staticmethod
    def replace_head(
        model_path: os.PathLike[str] | str,
        new_model_path: os.PathLike[str] | str,
        backbone_encoder: BackboneEncoder,
        transformer_params: TransformerParams,
    ) -> ActionRecognitionModel:
        old_model, num_samples = ActionRecognitionModel.load(
            Path(model_path),
            backbone_encoder,
        )
        backbone_encoder = old_model._encoder
        backbone_encoder.embed_dim = transformer_params.embed_dim
        new_model: ActionRecognitionModel = ActionRecognitionModel(
            encoder=old_model._encoder,
            transformer_params=transformer_params,
            T=old_model.T,
            stride=old_model.stride,
        )
        new_model.save(str(new_model_path), num_samples)
        return new_model


if __name__ == "__main__":
    from torchvision import transforms

    from ...retrieval.kinetics600.kinetics_600_retriever import K600Retriever
    from .movinet_a0s_encoder import MovinetEncoder

    T = 5
    stride = 5
    transformer_params = TransformerParams(
        embed_dim=480,
        depth=2,
        num_heads=16,
        mlp_dim=4 * 480,
        num_classes=2,
        head_dim=480 // 16,
    )
    transform = transforms.Compose(
        [
            lambda v: v.to(torch.float32) / 255,
            transforms.Resize((200, 200)),
            transforms.CenterCrop((172, 172)),
        ],
    )
    model = ActionRecognitionModel(
        MovinetEncoder(embed_dim=transformer_params.embed_dim),
        transformer_params,
        T=T,
        stride=stride,
    )
    model.eval()

    k600_retriever = K600Retriever.from_config(
        {
            "root": "/home/gil/data/k600",
            "frames_per_clip": 50,
            "frame_rate": 5,
            "positive_class_idx": 0,
        },
    )
    id_stream = k600_retriever._get_next_objectid()
    video_id = next(id_stream)
    video = k600_retriever.get_ml_data(video_id)

    with io.BytesIO(video.content) as f:
        video_content = torch.load(f)

    with torch.no_grad():
        X = transform(video_content).unsqueeze(dim=0)
        logits, _ = model(X)
    print(f"logits={logits.squeeze()}")
