# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import queue
import threading
import time
from abc import ABCMeta, abstractmethod
from typing import Iterable, Sized

from ...proto.messages_pb2 import HawkObject
from ..context.data_manager_context import DataManagerContext
from ..core.object_provider import ObjectProvider
from ..core.utils import get_server_ids

KEYS = ['total_objects', 'total_images', 'dropped_objects',
        'false_negatives', 'retrieved_images', 'retrieved_tiles']

class RetrieverStats(object):

    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])

class RetrieverBase(metaclass=ABCMeta):

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def get_objects(self) -> Iterable[ObjectProvider]:
        pass

    @abstractmethod
    def get_object(self, object_id: str, attributes: Sized) -> HawkObject:
        pass

    @abstractmethod
    def get_stats(self) -> RetrieverStats:
        pass

    @abstractmethod
    def is_running(self) -> bool: 
        pass

class Retriever(RetrieverBase):

    def __init__(self) -> None:
        self._context = None
        self._start_event = threading.Event()
        self._stop_event = threading.Event()
        self._command_lock = threading.RLock()
        self._stats = {x: 0 for x in KEYS}
        self._start_time = time.time()
        self.result_queue = queue.Queue()
        self.server_id = get_server_ids()[0]

    def start(self) -> None:
        with self._command_lock:
            self._start_time = time.time()

        self._start_event.set()
        threading.Thread(target=self.stream_objects, name='stream').start()

    def stop(self) -> None:
        self._stop_event.set()

    def is_running(self) -> bool: 
        return not self._stop_event.is_set()

    def add_context(self, context: DataManagerContext):
        self._context = context

    def stream_objects(self):
        # wait for mission context to be added
        while self._context is None:
            continue
        self._start_time = time.time()

    def set_tile_attributes(self, object_id: str, label: str):
        attributes = {
            'Device-Name': str.encode(self.server_id),
            '_ObjectID': str.encode(object_id),
            '_gt_label': str.encode(label),
        }
        return attributes

    def get_objects(self) -> Iterable[ObjectProvider]:
        return self.result_queue.get()

    def get_object(self, object_id: str, attributes: Sized = []) -> HawkObject:
        image_path = object_id.split("collection/id/")[-1]
        with open(image_path, 'rb') as f:
            content = f.read()

        # Return object attributes
        dct = {
                'Device-Name': str.encode(self.server_id),
                '_ObjectID': str.encode(object_id),
              }

        return HawkObject(objectId=object_id, content=content, attributes=dct)

    def get_stats(self) -> RetrieverStats:
        self._start_event.wait()

        with self._command_lock:
            stats = self._stats.copy()

        return RetrieverStats(stats)
