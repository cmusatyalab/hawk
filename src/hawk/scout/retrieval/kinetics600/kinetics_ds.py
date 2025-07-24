# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from torchvision.datasets import Kinetics

if TYPE_CHECKING:
    from torch import Tensor


class KineticsDs(Kinetics):  # type: ignore[misc]
    _metadata_path = Path("video_clips_cache", "train.pkl")

    def __init__(
        self,
        root: Path,
        frames_per_clip: int,
        frame_rate: int | None = None,
        step_between_clips: int = 1,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        metadata_cache = root / self._metadata_path
        with metadata_cache.open("rb") as file:
            _precomputed_metadata: dict[str, Any] | None = pickle.load(file)

        super().__init__(
            root=root,
            frames_per_clip=frames_per_clip,
            num_classes="600",
            split="train",
            frame_rate=frame_rate,
            step_between_clips=step_between_clips,
            transform=transform,
            _precomputed_metadata=_precomputed_metadata,
        )

    def get_video(self, idx: int) -> Tensor:
        video, _audio, _info, _index = self.video_clips.get_clip(idx)
        if self.transform is not None:
            video = self.transform(video)
        return video

    def get_label(self, idx: int) -> int:
        nclips = self.video_clips.num_clips()
        if idx >= nclips:
            msg = f"Index {idx} out of range ({nclips} number of clips)"
            raise IndexError(msg)

        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        label: int = self.samples[video_idx][1]
        return label

    @classmethod
    def download(cls, root: Path, num_download_workers: int = 1) -> None:
        print("Downloading Kinetics600 dataset to {root}.")

        ds = Kinetics(
            root=root,
            frames_per_clip=30,
            num_classes="600",
            split="train",
            download=True,
            num_download_workers=num_download_workers,
        )

        metadata_cache = root / cls._metadata_path
        with metadata_cache.open("wb") as file:
            pickle.dump(ds.video_clips.metadata, file)

        print("Kinetics600 dataset downloaded to {root}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Kinetics-600 dataset")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of download workers",
    )
    parser.add_argument("root", type=Path, help="Root directory")
    args = parser.parse_args()

    KineticsDs.download(args.root, num_download_workers=args.num_workers)
