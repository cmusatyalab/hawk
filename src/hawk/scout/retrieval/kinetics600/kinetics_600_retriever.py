import random
from typing import List, Union, Tuple, Dict
from pathlib import Path

from pandas import DataFrame
from torch import Tensor

from src.hawk.scout.retrieval.kinetics600.kinetics_ds import KineticsDs
from src.hawk.scout.retrieval.kinetics600.video_utils import create_gif_from_video_tensor_bytes
from src.hawk.scout.retrieval.retriever_ifc import RetrieverIfc


class K600Retriever(RetrieverIfc):
    def __init__(self, root: str, frames_per_clip, frame_rate: int, positive_class_idx: int = 0):
        self.frame_rate: int = frame_rate
        self.frames_per_clip: int = frames_per_clip
        self.root: str = root
        self.positive_class_idx: int = positive_class_idx
        label_transform = lambda y: 1 if y == positive_class_idx else 0
        self.ds = KineticsDs(root=root,
                             video_clips_pkl_name='train.pkl',
                             frames_per_clip=frames_per_clip,
                             step_between_clips=frames_per_clip,
                             split='train',
                             frame_rate=frame_rate,
                             label_transform=label_transform)

    def object_ids_stream(self):
        ## Generator - note that the videos order would be random
        ## and different at for each generator returned from this method.
        num_videos = len(self.ds)
        video_indexes = list(range(num_videos))
        random.shuffle(video_indexes)
        for video_index in video_indexes:
            yield video_index

    def get_ml_ready_data(self, object_ids: Union[List[int],int]) \
            -> Union[List[Tuple[Tensor,int]],Tuple[Tensor,int]]:
        if isinstance(object_ids, int):
            object_id = object_ids
            video, _ = self.ds[object_id]
            return video, object_id
        else:
            videos, indexes = [], []
            for object_id in object_ids:
                video, _ = self.ds[object_id]
                videos.append(video)
                indexes.append(object_id)
            return [(video,object_id) for video,object_id in zip(videos,indexes)]

    def get_oracle_ready_data(self, object_ids: Union[List[int],int]) \
            -> Union[List[Tuple[bytes,int]],Tuple[bytes,int]]:
        if isinstance(object_ids, int):
            object_id = object_ids
            video, _ = self.ds[object_id]
            gif = create_gif_from_video_tensor_bytes(video)
            return gif, object_id
        else:
            gifs, indexes = [], []
            for object_id in object_ids:
                video, _ = self.ds[object_id]
                gifs.append(create_gif_from_video_tensor_bytes(video))
                indexes.append(object_id)
            return [(gif,object_id) for gif,object_id in zip(gifs,indexes)]

    def get_ground_truth(self, object_ids: Union[List[Union[str, int]],Union[str, int]])\
            -> Union[List[Tuple[int,int]],Tuple[int,int]]:
        if isinstance(object_ids, int):
            object_id = object_ids
            label = self.ds.get_label(object_id)
            return label, object_id
        else:
            labels, indexes = [], []
            for object_id in object_ids:
                label = self.ds.get_label(object_id)
                labels.append(label)
                indexes.append(object_id)
            return [(label,object_id) for label,object_id in zip(labels,indexes)]

    def __len__(self):
        return len(self.ds)

    def generate_index_files(self, num_scouts: int, path: str) -> List[DataFrame]:
        assert num_scouts > 0
        shard_size = len(self) // num_scouts
        id_stream = self.object_ids_stream()
        res: List[DataFrame] = []
        for shard_id in range(num_scouts-1):
            scout_index: Dict[int,int] = dict()  # video_id -> label
            for _ in range(shard_size):
                video_id = next(id_stream)
                scout_index[video_id] = self.get_ground_truth(video_id)[0]
                assert scout_index[video_id] in {0,1}
            res.append(DataFrame.from_dict(scout_index, orient='index', columns=['label']))
        scout_index: Dict[int, int] = dict()
        for video_id in id_stream:
            scout_index[video_id] = self.get_ground_truth(video_id)[0]
            assert scout_index[video_id] in {0, 1}
        res.append(DataFrame.from_dict(scout_index, orient='index', columns=['label']))
        output_path = Path(path)
        for shard_id, shard in enumerate(res):
            shard.to_csv(f'{path}/scout_{shard_id}.csv')
            shard.to_csv(output_path / f'scout_{shard_id}.csv', index_label='video_id')
        return res



if __name__ == '__main__':
    k600_retriever = K600Retriever(root='/home/gil/data/k600',
                                  frames_per_clip=30,
                                  frame_rate=5,
                                  positive_class_idx=0)
    k600_retriever.generate_index_files(num_scouts=1, path='./')
    id_stream = k600_retriever.object_ids_stream()
    video_id = next(id_stream)
    video, id = k600_retriever.get_ml_ready_data(video_id)
    gif, id_  = k600_retriever.get_oracle_ready_data(video_id)
    assert id == id_
    with open("in_memory_output.gif", "wb") as f:
        f.write(gif)
    label, id = k600_retriever.get_ground_truth(video_id)
    print(f'label: {label}, id: {id}')