import os
import pickle
from typing import Optional, Callable, Dict, Any, Tuple

from torch import Tensor
from torchvision.datasets import Kinetics


class KineticsDs(Kinetics):


    def __init__(self,
                 root: str,
                 frames_per_clip: int,
                 video_clips_pkl_name: str,
                 split: str,
                 frame_rate: Optional[int] = None,
                 step_between_clips: int = 1,
                 transform: Optional[Callable] = None,
                 label_transform: Optional[Callable] = None,
                 ):
        video_clips_pkl_path = os.path.join(root, 'video_clips_cache', video_clips_pkl_name)
        with open(video_clips_pkl_path, 'rb') as file:
            _precomputed_metadata: Optional[Dict[str, Any]] = pickle.load(file)
        super().__init__(root=root,
                         frames_per_clip=frames_per_clip,
                         num_classes='600',
                         split=split,
                         frame_rate=frame_rate,
                         step_between_clips=step_between_clips,
                         transform=transform,
                         _precomputed_metadata=_precomputed_metadata)
        self.label_transform: Optional[Callable] = label_transform

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        video, _, __, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return video, label
