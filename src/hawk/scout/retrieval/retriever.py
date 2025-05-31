# SPDX-FileCopyrightText: 2022-2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import queue
import sys
import threading
import time
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from logzero import logger
from PIL import Image

from ...classes import (
    NEGATIVE_CLASS,
    ClassLabel,
    ClassName,
    class_label_to_int,
)
from ...hawkobject import HawkObject
from ...objectid import ObjectId
from ..context.data_manager_context import DataManagerContext
from ..core.result_provider import BoundingBox
from ..core.utils import get_server_ids
from ..stats import (
    HAWK_RETRIEVER_DROPPED_OBJECTS,
    HAWK_RETRIEVER_FAILED_OBJECTS,
    HAWK_RETRIEVER_QUEUE_LENGTH,
    HAWK_RETRIEVER_RETRIEVED_IMAGES,
    HAWK_RETRIEVER_RETRIEVED_OBJECTS,
    HAWK_RETRIEVER_TOTAL_IMAGES,
    HAWK_RETRIEVER_TOTAL_OBJECTS,
    collect_metrics_total,
)

THUMBNAIL_SIZE = (256, 256)


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
    def put_objectid(self, object_id: ObjectId) -> None:
        pass

    @abstractmethod
    def get_stats(self) -> RetrieverStats:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        pass

    @abstractmethod
    def get_ml_batch(
        self, batch_size: int, timeout: float | None = None
    ) -> tuple[list[ObjectId], list[HawkObject]]:
        """Get a batch of ML ready items."""
        pass

    @abstractmethod
    def get_ml_data(self, object_id: ObjectId) -> HawkObject | None:
        """Get ML ready tile for inferencing or training.

        returns None if the object_id is not found.
        """
        pass

    @abstractmethod
    def get_oracle_data(self, object_id: ObjectId) -> list[HawkObject]:
        """Get Oracle ready data to for labeling at home.

        raises FileNotFoundError if the object_id is not found.
        raises ValueError if we failed to create oracle ready data.
        """
        pass

    @abstractmethod
    def get_groundtruth(self, object_id: ObjectId) -> list[BoundingBox]:
        """Get groundtruth for logging, statistics and scriptlabeler."""
        pass


class Retriever(RetrieverBase):
    def __init__(
        self,
        mission_id: str,
        tiles_per_interval: int = 200,
        globally_constant_rate: bool = False,
    ) -> None:
        self._context: DataManagerContext | None = None
        self._context_event = threading.Event()
        self._start_event = threading.Event()
        self._stop_event = threading.Event()
        self._command_lock = threading.RLock()
        self._start_time = time.time()
        self.result_queue: queue.Queue[ObjectId] = queue.Queue()
        self.server_id = get_server_ids()[0]
        self.total_tiles = 0

        # TODO: we probably should be more restrictive about where images
        # are allowed to be retrieved from.
        # Random retriever sets this to the parent of the INDEXES directory.
        self._data_root = Path("/")

        # These parameters control the rate at which tiles are retrieved and
        # inferenced at the scout. If we want to achieve a globally constant
        # rate during a SCML scenario, we periodically update the active scout
        # ratio and only process `tiles_per_interval / active_scout_ratio`
        # tiles at a time so that when scouts are added or lost, the remaining
        # scouts adjust their processing rate to compensate.
        # Currently only used by network_retriever
        self.tiles_per_interval: int = tiles_per_interval
        self.active_scout_ratio: float = 1.0
        self.globally_constant_rate: bool = globally_constant_rate

        self.current_deployment_mode = "Idle"
        self.scml_active_mode: bool | None = None

        self.total_images = HAWK_RETRIEVER_TOTAL_IMAGES.labels(mission=mission_id)
        self.total_objects = HAWK_RETRIEVER_TOTAL_OBJECTS.labels(mission=mission_id)
        self.retrieved_images = HAWK_RETRIEVER_RETRIEVED_IMAGES.labels(
            mission=mission_id
        )
        self.retrieved_objects = HAWK_RETRIEVER_RETRIEVED_OBJECTS.labels(
            mission=mission_id
        )
        self.failed_objects = HAWK_RETRIEVER_FAILED_OBJECTS.labels(mission=mission_id)
        self.dropped_objects = HAWK_RETRIEVER_DROPPED_OBJECTS.labels(mission=mission_id)
        self.queue_length = HAWK_RETRIEVER_QUEUE_LENGTH.labels(mission=mission_id)

    def start(self) -> None:
        with self._command_lock:
            self._start_time = time.time()

        self._start_event.set()
        self._run_threads()

    def _run_threads(self) -> None:
        """start default thread for all retrievers and network clients
        can be overridden in derived classes such as the network_retriever."""
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
        self.current_deployment_mode = "Active"

    def server(self) -> None:
        ## placeholder for server in network retriever
        pass

    def put_objectid(self, result_object: ObjectId, dup: bool = False) -> None:
        if not dup:
            self.retrieved_objects.inc()
        try:
            self.queue_length.inc()
            self.result_queue.put_nowait(result_object)
        except queue.Full:
            self.queue_length.dec()
            self.dropped_objects.inc()

    def get_objectid(self, timeout: float | None = None) -> ObjectId | None:
        """Get a single ML ready object identifier.

        Returns None when there was no candidate before the timeout expired.
        """
        try:
            object_id = self.result_queue.get(timeout=timeout)
            self.queue_length.dec()
            return object_id
        except queue.Empty:
            return None

    def get_stats(self) -> RetrieverStats:
        self._start_event.wait()

        stats = RetrieverStats(
            total_objects=collect_metrics_total(self.total_objects)
            - collect_metrics_total(self.failed_objects),
            total_images=collect_metrics_total(self.total_images),
            dropped_objects=collect_metrics_total(self.dropped_objects),
            retrieved_images=collect_metrics_total(self.retrieved_images),
            retrieved_tiles=collect_metrics_total(self.retrieved_objects),
        )
        return stats

    def _class_id_to_name(self, class_label: ClassLabel) -> ClassName:
        # making sure the negative class is always called "negative" to make
        # graphing easier
        if class_label == ClassLabel(0):
            return NEGATIVE_CLASS
        if self._context is None:
            class_index = class_label_to_int(class_label)
            return ClassName(sys.intern(f"class-{class_index}"))
        return self._context.class_list[class_label]

    def get_ml_batch(
        self, batch_size: int, timeout: float | None = None
    ) -> tuple[list[ObjectId], list[HawkObject]]:
        """Get a batch of ML ready items."""
        object_ids: list[ObjectId] = []
        hawk_objects: list[HawkObject] = []

        time_now = time.time()
        wait_time: float | None = None
        end_time = time_now + timeout if timeout is not None else None

        while len(hawk_objects) < batch_size:
            if end_time is not None:
                # we use time_now so that a timeout of 0 will still end up
                # trying to get something off the queue (get_nowait)
                wait_time = end_time - time_now
                if wait_time < 0:
                    break

            object_id = self.get_objectid(timeout=wait_time)
            if object_id is None:
                break

            data = self.get_ml_data(object_id)
            if data is not None:
                object_ids.append(object_id)
                hawk_objects.append(data)

            # we have to make sure to update time_now before we loop
            time_now = time.time()

        return (object_ids, hawk_objects)

    def get_ml_data(self, object_id: ObjectId) -> HawkObject | None:
        """Get ML ready tile for inferencing or training."""
        object_path = object_id._file_path(self._data_root)
        if object_path is None:
            logger.error(f"Unable to get path for {object_id}")
            return None
        try:
            return HawkObject.from_file(object_path)
        except FileNotFoundError:
            logger.error(f"Unable to read {object_id}")
            return None

    def get_oracle_data(self, object_id: ObjectId) -> list[HawkObject]:
        """Get Oracle ready data to for labeling at home."""
        ml_object = self.get_ml_data(object_id)
        if ml_object is None or not ml_object.media_type.startswith("image/"):
            raise ValueError("Generic get_oracle_data only works for images")

        with BytesIO(ml_object.content) as f:
            image = Image.open(f)

        image = image.convert("RGB")

        # crop to centered square
        if image.size[0] != image.size[1]:
            short_edge = min(image.size)
            left = (image.size[0] - short_edge) // 2
            top = (image.size[1] - short_edge) // 2
            right = left + short_edge
            bottom = top + short_edge
            image = image.crop((left, top, right, bottom))

        # resize to THUMBNAIL_SIZE
        # image.thumbnail(THUMBNAIL_SIZE)
        image = image.resize(THUMBNAIL_SIZE)

        with BytesIO() as tmpfile:
            image.save(tmpfile, format="JPEG", quality=85)
            content = tmpfile.getvalue()

        return [HawkObject(content=content, media_type="image/jpeg")]

    def get_groundtruth(self, object_id: ObjectId) -> list[BoundingBox]:
        """Get groundtruth for logging, statistics and scriptlabeler."""
        # only handles classification groundtruth for now and gets it the wrong
        # way by assuming it is stashed in the object id.
        class_name = object_id._groundtruth()
        if class_name is None:
            return []

        return [
            BoundingBox(
                x=0.5, y=0.5, w=1.0, h=1.0, class_name=class_name, confidence=1.0
            )
        ]
