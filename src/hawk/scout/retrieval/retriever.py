# SPDX-FileCopyrightText: 2022-2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import queue
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterator

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from typing_extensions import Self

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


@dataclass
class RetrieverStats:
    total_objects: int = 0
    total_images: int = 0
    dropped_objects: int = 0
    retrieved_images: int = 0
    retrieved_tiles: int = 0


class EnvOverrideConfig(BaseSettings):
    # reorder initialization order so that environment vars (admin configured)
    # override init settings (user provided).
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            dotenv_settings,
            file_secret_settings,
            init_settings,
        )


class RetrieverConfig(EnvOverrideConfig):
    """Base config class for Retrievers."""

    model_config = SettingsConfigDict(
        env_prefix="hawk_retriever_",
        env_file="hawk.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # should be set by the admin to restrict which filesystem subtree can be
    # accessed, i.e.
    #   export HAWK_RETRIEVER_DATA_ROOT=/path/to/data
    data_root: Path = Field(default_factory=Path.cwd, validate_default=False)

    # uniquely identifies the current mission, used for logging/stats/etc.
    mission_id: str


class ImageRetrieverConfig(RetrieverConfig):
    """Common config class for image retrievers.

    Used when a retriever returns images from get_ml_data, and the user may
    request images to be cropped and/or resized before inferencing.
    """

    resize_tile: bool = False
    tile_size: int = 256


class RetrieverBase(ABC):
    config_class: ClassVar[type[RetrieverConfig]]
    config: RetrieverConfig

    @abstractmethod
    def get_next_objectid(self) -> Iterator[ObjectId]:
        """Iterator yielding object ids."""

    @abstractmethod
    def get_ml_data(self, object_id: ObjectId) -> HawkObject:
        """Get ML ready tile for inferencing or training.

        raise FileNotFoundError if the object_id is not found.
        """

    @abstractmethod
    def get_oracle_data(self, object_id: ObjectId) -> list[HawkObject]:
        """Get Oracle ready data to for labeling at home.

        raise FileNotFoundError if the object_id is not found.
        raise ValueError if we fail to create oracle ready data.
        """

    @abstractmethod
    def get_groundtruth(self, object_id: ObjectId) -> list[BoundingBox]:
        """Get groundtruth for logging, statistics and script labeler."""


class Retriever(RetrieverBase):
    ## Validators and constructors

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> RetrieverConfig:
        return cls.config_class.model_validate(
            {k: v for k, v in config.items() if not k.startswith("_")}
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        return cls(cls.validate_config(config))

    def __init__(self, config: RetrieverConfig) -> None:
        self.config = config

        self._context: DataManagerContext | None = None
        self._context_event = threading.Event()
        self._start_event = threading.Event()
        self._stop_event = threading.Event()
        self._command_lock = threading.RLock()
        self._start_time = time.time()
        self.result_queue: queue.Queue[ObjectId] = queue.Queue()
        self.server_id = get_server_ids()[0]
        self.total_tiles = 0

        self.current_deployment_mode = "Idle"
        self.scml_active_mode: bool | None = None

        self.total_images = HAWK_RETRIEVER_TOTAL_IMAGES.labels(
            mission=self.config.mission_id
        )
        self.total_objects = HAWK_RETRIEVER_TOTAL_OBJECTS.labels(
            mission=self.config.mission_id
        )
        self.retrieved_images = HAWK_RETRIEVER_RETRIEVED_IMAGES.labels(
            mission=self.config.mission_id
        )
        self.retrieved_objects = HAWK_RETRIEVER_RETRIEVED_OBJECTS.labels(
            mission=self.config.mission_id
        )
        self.failed_objects = HAWK_RETRIEVER_FAILED_OBJECTS.labels(
            mission=self.config.mission_id
        )
        self.dropped_objects = HAWK_RETRIEVER_DROPPED_OBJECTS.labels(
            mission=self.config.mission_id
        )
        self.queue_length = HAWK_RETRIEVER_QUEUE_LENGTH.labels(
            mission=self.config.mission_id
        )

    def add_context(self, context: DataManagerContext) -> None:
        self._context = context
        self._context_event.set()

    def start(self) -> None:
        with self._command_lock:
            self._start_time = time.time()

        self._start_event.set()
        self._run_threads()

    def stop(self) -> None:
        self._stop_event.set()

    def _run_threads(self) -> None:
        """start default thread for all retrievers and network clients
        can be overridden in derived classes such as the network_retriever."""
        threading.Thread(target=self._stream_objects, name="stream").start()

    def _stream_objects(self) -> None:
        # wait for mission context to be added
        while self._context is None:
            self._context_event.wait()
        self._start_time = time.time()
        self.current_deployment_mode = "Active"

        for object_id in self.get_next_objectid():
            if self._stop_event.is_set():
                break
            self.put_objectid(object_id)

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
        """Pop a single ML ready object identifier from the result queue."""
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
