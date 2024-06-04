# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import dataclasses
import queue
import threading
import time
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ...proto.messages_pb2 import HawkObject
from ..context.data_manager_context import DataManagerContext
from ..core.object_provider import ObjectProvider
from ..core.utils import get_server_ids


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
    def get_object(self, object_id: str) -> Optional[HawkObject]:
        pass

    @abstractmethod
    def get_stats(self) -> RetrieverStats:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        pass


class Retriever(RetrieverBase):
    def __init__(self) -> None:
        self._context: Optional[DataManagerContext] = None
        self._context_event = threading.Event()
        self._start_event = threading.Event()
        self._stop_event = threading.Event()
        self._command_lock = threading.RLock()
        self._stats = RetrieverStats()
        self._start_time = time.time()
        self.result_queue: queue.Queue[ObjectProvider] = queue.Queue()
        self.server_id = get_server_ids()[0]
        self.total_tiles = 0

    def start(self) -> None:
        with self._command_lock:
            self._start_time = time.time()

        self._start_event.set()
        threading.Thread(target=self.stream_objects, name="stream").start()

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

    def set_tile_attributes(self, object_id: str, label: str) -> Dict[str, bytes]:
        attributes = {
            "Device-Name": str.encode(self.server_id),
            "_ObjectID": str.encode(object_id),
            "_gt_label": str.encode(label),
        }
        return attributes

    def get_objects(self) -> ObjectProvider:
        return self.result_queue.get()

    def put_objects(self, result_object) -> None:
        self.result_queue.put_nowait(result_object)

    def get_object(self, object_id: str) -> Optional[HawkObject]:
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

        with self._command_lock:
            stats = dataclasses.replace(self._stats)

        return stats
