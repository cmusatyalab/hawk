# Copyright (c) 2024 Carnegie Mellon University
# SPDX-License-Identifier: GPLv2-only

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import pytest
from PIL import Image

from hawk import HawkObject, ObjectId
from hawk.scout.context.model_trainer_context import ModelContext
from hawk.scout.retrieval.retriever import Retriever, RetrieverConfig

if TYPE_CHECKING:
    from hawk import Detection
    from hawk.scout.core.hawk_stub import HawkStub


REFERENCE_IMAGE = Path(__file__).parent.parent.joinpath(
    "assets",
    "grace_hopper_517x606.jpg",
)


def test_hawkobject() -> HawkObject:
    return HawkObject(content=REFERENCE_IMAGE.read_bytes(), media_type="image/jpeg")


class TestRetriever(Retriever):
    config_class = RetrieverConfig

    def get_next_objectid(self) -> Iterator[ObjectId]:
        yield ObjectId(oid="test_oid")

    def get_ml_data(self, object_id: ObjectId) -> HawkObject:
        return test_hawkobject()

    def get_oracle_data(self, object_id: ObjectId) -> list[HawkObject]:
        return [test_hawkobject()]

    def get_groundtruth(self, object_id: ObjectId) -> list[Detection]:
        return []


class TestContext(ModelContext):
    @property
    def scout_index(self) -> int:
        return 0

    @property
    def scouts(self) -> list[HawkStub]:
        return []

    @property
    def port(self) -> int:
        return 0

    @property
    def model_dir(self) -> Path:
        return Path.cwd()

    def stop_model(self) -> None:
        pass

    def check_create_test(self) -> bool:
        return False


@pytest.fixture
def objectid() -> ObjectId:
    return ObjectId("/negative/collection/id/test-objectid")


@pytest.fixture
def testretriever() -> TestRetriever:
    return TestRetriever.from_config({"mission_id": "test-mission"})


@pytest.fixture
def testcontext(testretriever: TestRetriever) -> TestContext:
    return TestContext(retriever=testretriever)


@pytest.fixture
def reference_image() -> Image.Image:
    return Image.open(REFERENCE_IMAGE).convert("RGB")
