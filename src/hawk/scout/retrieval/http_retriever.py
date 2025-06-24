# type: ignore

# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import time
from itertools import count
from typing import Iterator

import httpx
from logzero import logger
from pydantic import HttpUrl

from ...hawkobject import HawkObject
from ...objectid import ObjectId
from ..core.result_provider import BoundingBox
from ..stats import collect_metrics_total
from .retriever import Retriever, RetrieverConfig


class HTTPRetrieverConfig(RetrieverConfig):
    base_url: HttpUrl  # base url of the server
    timeout: float = 20.0  # the rate at which frames (batch of tiles) are returned
    tiles_per_frame: int = 200  # tiles per image


class HTTPRetriever(Retriever):
    config_class = HTTPRetrieverConfig
    config: HTTPRetrieverConfig

    def __init__(self, config: HTTPRetrieverConfig) -> None:
        super().__init__(config)

        self.http = httpx.Client(
            base_url=f"{self.config.base_url}/{self.config.mission_id}"
        )

        self.total_images.set(0)
        self.total_objects.set(0)

    def get_next_objectid(self) -> Iterator[ObjectId]:
        time_start = time.time()
        for ntiles in count(1):
            r = self.http.get("/get_next_oid")
            if r.status_code != httpx.codes.OK:
                break

            oid = r.json()["oid"]
            logger.debug(f"/get_next_oid elapsed: {r.elapsed.total_seconds()}")

            self.total_objects.inc()

            yield ObjectId(oid)

            retrieved_tiles = collect_metrics_total(self.retrieved_objects)
            logger.info(f"{retrieved_tiles} RETRIEVED")

            if ntiles % self.config.tiles_per_frame != 0:
                continue

            self.total_images.inc()

            time_passed = time.time() - time_start
            if time_passed < self.config.timeout:
                time.sleep(self.config.timeout - time_passed)
            time_start = time.time()

    def get_ml_data(self, object_id: ObjectId) -> HawkObject:
        oid = object_id.serialize_oid()
        r = self.http.get(f"/get_ml_data/{oid}").raise_for_status()
        obj = HawkObject(content=r.content, media_type=r.headers["content-type"])
        logger.debug(f"/get_ml_data/{oid} elapsed: {r.elapsed.total_seconds()}")
        return obj

    def get_oracle_data(self, object_id: ObjectId) -> list[HawkObject]:
        oid = object_id.serialize_oid()
        oracle_data = []
        for i in range(10):
            r = self.http.get(f"/get_oracle_data/{i}/{oid}")
            if r.status_code != httpx.codes.OK:
                break

            obj = HawkObject(content=r.content, media_type=r.headers["content-type"])
            logger.debug(
                f"/get_oracle_data/{i}/{oid} elapsed: {r.elapsed.total_seconds()}"
            )
            oracle_data.append(obj)
        return oracle_data

    def get_groundtruth(self, object_id: ObjectId) -> list[BoundingBox]:
        oid = object_id.serialize_oid()
        r = httpx.get(f"/get_groundtruth/{oid}").raise_for_status()
        # ...decode json reponse to list of boundingboxes/detections...
        logger.debug(f"/get_groundtruth/{oid} elapsed: {r.elapsed.total_seconds()}")
        return []
