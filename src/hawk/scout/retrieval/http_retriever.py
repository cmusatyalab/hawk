# type: ignore

# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from itertools import count
from typing import TYPE_CHECKING, Iterator

import httpx
from logzero import logger

from ...hawkobject import HawkObject
from ...objectid import ObjectId
from .retriever import Retriever, RetrieverConfig

if TYPE_CHECKING:
    from pydantic import HttpUrl

    from ...detection import Detection


class HTTPRetrieverConfig(RetrieverConfig):
    base_url: HttpUrl  # base url of the server
    tiles_per_frame: int = 200  # tiles per image


class HTTPRetriever(Retriever):
    config_class = HTTPRetrieverConfig
    config: HTTPRetrieverConfig

    def __init__(self, config: HTTPRetrieverConfig) -> None:
        super().__init__(config)

        self.http = httpx.Client(
            base_url=f"{self.config.base_url}/{self.config.mission_id}",
        )

        self.total_images.set(0)
        self.total_objects.set(0)

    def get_next_objectid(self) -> Iterator[ObjectId | None]:
        for ntiles in count(1):
            r = self.http.get("/get_next_oid")
            if r.status_code != httpx.codes.OK:
                break

            oid = r.json()["oid"]
            logger.debug(f"/get_next_oid elapsed: {r.elapsed.total_seconds()}")

            self.total_objects.inc()

            yield ObjectId(oid)

            if ntiles % self.config.tiles_per_frame == 0:
                yield None

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
                f"/get_oracle_data/{i}/{oid} elapsed: {r.elapsed.total_seconds()}",
            )
            oracle_data.append(obj)
        return oracle_data

    def get_groundtruth(self, object_id: ObjectId) -> list[Detection]:
        oid = object_id.serialize_oid()
        r = httpx.get(f"/get_groundtruth/{oid}").raise_for_status()
        # ...decode json response to list of boundingboxes/detections...
        logger.debug(f"/get_groundtruth/{oid} elapsed: {r.elapsed.total_seconds()}")
        return []
