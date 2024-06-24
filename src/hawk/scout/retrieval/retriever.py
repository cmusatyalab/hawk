# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import queue
import threading
import time
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from logzero import logger

from ...proto.messages_pb2 import HawkObject
from ..context.data_manager_context import DataManagerContext
from ..core.object_provider import ObjectProvider
from ..core.utils import get_server_ids
from ..stats import (
    HAWK_RETRIEVER_DROPPED_OBJECTS,
    HAWK_RETRIEVER_QUEUE_LENGTH,
    HAWK_RETRIEVER_RETRIEVED_IMAGES,
    HAWK_RETRIEVER_RETRIEVED_OBJECTS,
    HAWK_RETRIEVER_TOTAL_IMAGES,
    HAWK_RETRIEVER_TOTAL_OBJECTS,
    collect_metrics_total,
)


@dataclass
class RetrieverStats:
    total_objects: int = 0
    total_images: int = 0
    dropped_objects: int = 0
    retrieved_images: int = 0
    retrieved_tiles: int = 0


class RetrieverBase(metaclass=ABCMeta):
    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def get_objects(self) -> ObjectProvider:
        pass

    @abstractmethod
    def put_objects(self, obj: ObjectProvider) -> None:
        pass

    @abstractmethod
    def read_object(self, object_id: str) -> Optional[HawkObject]:
        pass

    @abstractmethod
    def get_stats(self) -> RetrieverStats:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        pass


class Retriever(RetrieverBase):
    def __init__(self, mission_id: str) -> None:
        self._context: Optional[DataManagerContext] = None
        self._context_event = threading.Event()
        self._start_event = threading.Event()
        self._stop_event = threading.Event()
        self._command_lock = threading.RLock()
        self._start_time = time.time()
        self.result_queue: queue.Queue[ObjectProvider] = queue.Queue()
        self.server_id = get_server_ids()[0]
        self.total_tiles = 0

        self.total_images = HAWK_RETRIEVER_TOTAL_IMAGES.labels(mission=mission_id)
        self.total_objects = HAWK_RETRIEVER_TOTAL_OBJECTS.labels(mission=mission_id)
        self.retrieved_images = HAWK_RETRIEVER_RETRIEVED_IMAGES.labels(
            mission=mission_id
        )
        self.retrieved_objects = HAWK_RETRIEVER_RETRIEVED_OBJECTS.labels(
            mission=mission_id
        )
        self.dropped_objects = HAWK_RETRIEVER_DROPPED_OBJECTS.labels(mission=mission_id)
        self.queue_length = HAWK_RETRIEVER_QUEUE_LENGTH.labels(mission=mission_id)

    def start(self) -> None:
        with self._command_lock:
            self._start_time = time.time()

        self._start_event.set()
        if self.network and self._network_server_host == self.this_host_name:
            logger.info("Starting server thread in retriever...")
            threading.Thread(
                target=self.server, name="network_server"
            ).start()  ## launch server thread.
        else:
            threading.Thread(
                target=self.stream_objects, name="stream"
            ).start()  ## start default thread for all retrievers and network clients

    def stop(self) -> None:
        self._stop_event.set()

    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    def add_context(self, context: DataManagerContext) -> None:
        self._context = context
        self._context_event.set()

    def stream_objects(self) -> None:
        # wait for mission context to be added
        while self._context is None:
            self._context_event.wait()
        self._start_time = time.time()

    def server(self) -> None:
        ## placeholder for server in network retriever
        pass

    def set_tile_attributes(self, object_id: str, label: str) -> Dict[str, bytes]:
        attributes = {
            "Device-Name": str.encode(self.server_id),
            "_ObjectID": str.encode(object_id),
            "_gt_label": str.encode(label),
        }
        return attributes

    def get_objects(self) -> ObjectProvider:
        result = self.result_queue.get()
        self.queue_length.dec()
        return result

    def put_objects(self, result_object: ObjectProvider) -> None:
        self.retrieved_objects.inc()
        try:
            self.queue_length.inc()
            self.result_queue.put_nowait(result_object)
        except queue.Full:
            self.queue_length.dec()
            self.dropped_objects.inc()

    def read_object(self, object_id: str) -> Optional[HawkObject]:
        image_path = Path(object_id.split("collection/id/")[-1])
        if not image_path.exists():
            return None

        content = image_path.read_bytes()

        # Return object attributes
        dct = {
            "Device-Name": str.encode(self.server_id),
            "_ObjectID": str.encode(object_id),
        }
        return HawkObject(objectId=object_id, content=content, attributes=dct)

    def get_stats(self) -> RetrieverStats:
        self._start_event.wait()

        stats = RetrieverStats(
            total_objects=collect_metrics_total(self.total_objects),
            total_images=collect_metrics_total(self.total_images),
            dropped_objects=collect_metrics_total(self.dropped_objects),
            retrieved_images=collect_metrics_total(self.retrieved_images),
            retrieved_tiles=collect_metrics_total(self.retrieved_objects),
        )
        return stats
