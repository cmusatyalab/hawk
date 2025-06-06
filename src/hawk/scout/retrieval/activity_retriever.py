# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import time
from pathlib import Path

from logzero import logger

from .kinetics600.kinetics_600_retriever_helper import K600RetrieverHelper
from .retriever import Retriever
from ..core.attribute_provider import HawkAttributeProvider
from ..core.object_provider import ObjectProvider
from ..stats import collect_metrics_total
from ...classes import ClassLabel
from ...objectid import ObjectId
from ...proto.messages_pb2 import FileDataset


class ActivityRetriever(Retriever):
    def __init__(self, mission_id: str, dataset: FileDataset):
        super().__init__(mission_id)
        self._dataset = dataset
        self._timeout = dataset.timeout
        self._N: int = dataset.N
        self._fps: int = dataset.fps
        logger.info("In ACTIVITY RETRIEVER INIT...")

        index_file = Path(self._dataset.dataPath)
        self._data_root = index_file.parent
        self.k600_retriever: K600RetrieverHelper = K600RetrieverHelper(root=str(self._data_root),
                                             frames_per_clip=self._N,
                                             frame_rate=self._fps,
                                             positive_class_idx=0)

    def stream_objects(self) -> None:
        super().stream_objects()
        assert self._context is not None
        object_id_stream = self.k600_retriever.object_ids_stream()
        for video_id in object_id_stream:
            # self.put_objects(ObjectProvider(object_id, b"", None, None))
            time_start = time.time()
            if self._stop_event.is_set():
                break

            self.retrieved_images.inc()
            elapsed = time.time() - self._start_time
            logger.info(f"Retrieved Video:{video_id}  @ {elapsed}")

            content, _ = self.k600_retriever.get_ml_ready_data(video_id)
            y, _ = self.k600_retriever.get_ground_truth(video_id)
            class_label = ClassLabel(y)
            class_name = self._class_id_to_name(class_label)

            object_id = ObjectId(f"/{class_name}/collection/id/{video_id}")

            image_path = self._data_root / str(video_id)

            attributes = self.set_tile_attributes(object_id, class_name)

            self.put_objects(
                ObjectProvider(
                    object_id,
                    content,
                    HawkAttributeProvider(attributes, image_path, False,
                                          thumbnail=self.k600_retriever.get_oracle_ready_data(video_id)[0]),
                    class_name,
                )
            )

            retrieved_tiles = collect_metrics_total(self.retrieved_objects)
            logger.info(f"{retrieved_tiles} / {self.total_tiles} RETRIEVED")
            time_passed = time.time() - time_start
            if time_passed < self._timeout:
                time.sleep(self._timeout - time_passed)
