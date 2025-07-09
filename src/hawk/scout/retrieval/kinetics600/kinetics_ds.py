# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable

from torch import Tensor
from torchvision.datasets import Kinetics


class KineticsDs(Kinetics):  # type: ignore[misc]
    def __init__(
        self,
        root: Path,
        frames_per_clip: int,
        video_clips_pkl_name: str,
        split: str,
        frame_rate: int | None = None,
        step_between_clips: int = 1,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        video_clips_pkl_path = root / "video_clips_cache" / video_clips_pkl_name
        with open(video_clips_pkl_path, "rb") as file:
            _precomputed_metadata: dict[str, Any] | None = pickle.load(file)

        super().__init__(
            root=str(root),
            frames_per_clip=frames_per_clip,
            num_classes="600",
            split=split,
            frame_rate=frame_rate,
            step_between_clips=step_between_clips,
            transform=transform,
            _precomputed_metadata=_precomputed_metadata,
        )

    def __getitem__(self, idx: int) -> Tensor:
        video = self.video_clips.get_clip(idx)[0]
        if self.transform is not None:
            video = self.transform(video)
        return video

    def get_label(self, idx: int) -> int:
        nclips = self.video_clips.num_clips()
        if idx >= nclips:
            raise IndexError(f"Index {idx} out of range ({nclips} number of clips)")

        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        # video_path = self.video_clips.video_paths[video_idx]

        label: int = self.samples[video_idx][1]
        return label
