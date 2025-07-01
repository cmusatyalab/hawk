# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import io  # Import the io module
import os
import pickle
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.folder import find_classes, make_dataset
from torchvision.datasets.video_utils import VideoClips


class StoreMetadataVideoClips(VideoClips):  # type: ignore[misc]
    def __init__(
        self,
        metadata_path: str,
        video_paths: list[str],
        clip_length_in_frames: int,
        frames_between_clips: int,
        frame_rate: int | None = None,
        _precomputed_metadata: dict[str, Any] | None = None,
        num_workers: int = 0,
        output_format: str = "THWC",
    ):
        self.metadata_path: str = metadata_path
        super().__init__(
            video_paths=video_paths,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            frame_rate=frame_rate,
            _precomputed_metadata=_precomputed_metadata,
            num_workers=num_workers,
            output_format=output_format,
        )

    def _compute_frame_pts(self) -> None:
        super()._compute_frame_pts()
        metadata = self.metadata
        with open(self.metadata_path, "wb") as f:
            pickle.dump(metadata, f)


def calculate_metadata(
    pkl_name: str,
    root: str,
    frames_per_clip: int,
    split: str,
    step_between_clips: int,
    extensions: tuple[str, ...],
    frame_rate: int | None = None,
    num_workers: int = 1,
    output_format: str = "TCHW",
) -> None:
    split_folder = os.path.join(root, split)
    classes, class_to_idx = find_classes(split_folder)
    samples = make_dataset(split_folder, class_to_idx, extensions, is_valid_file=None)
    video_list = [x[0] for x in samples]
    StoreMetadataVideoClips(
        metadata_path=os.path.join(root, "video_clips_cache", pkl_name),
        video_paths=video_list,
        clip_length_in_frames=frames_per_clip,
        frames_between_clips=step_between_clips,
        frame_rate=frame_rate,
        num_workers=num_workers,
        output_format=output_format,
    )


def create_gif_from_video_tensor_bytes(video: torch.Tensor, fps: int = 10) -> bytes:
    """
    Creates a GIF from a sequence of RGB video frames represented as a PyTorch tensor
    and returns it as a binary (bytes) object.

    Args:
        video (torch.Tensor): A PyTorch tensor of shape (T, C, H, W) where:
                              T = number of frames
                              C = number of channels (RGB, so C=3)
                              H = height
                              W = width
                              Pixel values are integers in the range 0-255.
        fps (int, optional): Frames per second for the GIF. Defaults to 10.

    Returns:
        bytes: The GIF content as a binary object.
    """
    if not isinstance(video, torch.Tensor):
        raise TypeError("Input 'video' must be a PyTorch tensor.")
    if video.dim() != 4:
        raise ValueError("Input 'video' tensor must have 4 dimensions (T, C, H, W).")
    if video.shape[1] != 3:
        raise ValueError("Input 'video' tensor must have 3 channels (RGB).")
    if not torch.all((video >= 0) & (video <= 255)):
        print(
            "Warning: Pixel values are not strictly within 0-255. Clamping will occur."
        )

    # Convert PyTorch tensor to NumPy array
    # Permute dimensions from (T, C, H, W) to (T, H, W, C)
    frames_np = video.permute(0, 2, 3, 1).cpu().numpy()

    # Ensure data type is uint8 (unsigned 8-bit integer) and values are clamped
    frames_np = np.clip(frames_np, 0, 255).astype(np.uint8)

    # Save as animated gif with Pillow, use io.BytesIO as the in-memory file
    imgs = [Image.fromarray(img) for img in frames_np]
    with io.BytesIO() as byte_stream:
        imgs[0].save(
            byte_stream,
            save_all=True,
            append_images=imgs[1:],
            duration=1000 // fps,
            loop=1,
        )
        content = byte_stream.getvalue()
    return content


# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy video tensor
    T, C, H, W = 15, 3, 80, 100
    video_tensor = torch.zeros((T, C, H, W), dtype=torch.uint8)

    for i in range(T):
        # Create a simple animation: colors shifting
        r_val = int(255 * (i / (T - 1)))
        g_val = int(255 * ((T - 1 - i) / (T - 1)))
        b_val = int(128 + 127 * np.sin(i * np.pi / (T - 1)))  # Oscillating blue

        video_tensor[i, 0, :, :] = r_val
        video_tensor[i, 1, :, :] = g_val
        video_tensor[i, 2, :, :] = b_val

    # Get the GIF as bytes
    try:
        gif_data_bytes = create_gif_from_video_tensor_bytes(video_tensor, fps=8)

        if gif_data_bytes:
            print(f"GIF created in memory. Size: {len(gif_data_bytes)} bytes")

            # You can now do anything with `gif_data_bytes`, e.g.:
            # 1. Send it over a network (e.g., in a web API response)
            # 2. Embed it in a data URL for display in HTML
            #    (e.g., in a Jupyter Notebook)
            # 3. Save it to a file programmatically (if you decide to later)

            # Example: Save the bytes to a file (optional, just to verify it works)
            with open("in_memory_output.gif", "wb") as f:
                f.write(gif_data_bytes)
            print(
                "In-memory GIF also saved to 'in_memory_output.gif' for verification."
            )

    except Exception as e:
        print(f"An error occurred during GIF creation: {e}")

if __name__ == "__main__":
    calculate_metadata(
        pkl_name="train.pkl",
        root="/home/gil/data/k600/",
        split="train",
        step_between_clips=50,
        num_workers=8,
        frames_per_clip=50,
        frame_rate=5,
        extensions=("avi", "mp4"),
    )
