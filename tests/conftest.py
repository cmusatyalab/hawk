# Copyright (c) 2024 Carnegie Mellon University
# SPDX-License-Identifier: GPLv2-only

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from hawk.scout.context.model_trainer_context import ModelContext
from hawk.scout.core.attribute_provider import AttributeProvider
from hawk.scout.core.object_provider import ObjectProvider

if TYPE_CHECKING:
    from hawk.scout.core.hawk_stub import HawkStub


REFERENCE_IMAGE = Path(__file__).parent.joinpath("assets", "grace_hopper_517x606.jpg")


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
    def model_dir(self) -> int:
        return Path.cwd()

    def stop_model(self) -> None:
        pass

    def check_create_test(self) -> bool:
        return False


class TestAttributes(AttributeProvider):
    def get_image(self) -> Image.Image:
        return Image.open(REFERENCE_IMAGE).convert("RGB")

    def get_thumbnail_size(self) -> int:
        return 0

    def get(self) -> dict[str, bytes]:
        return dict()

    def add(self, attribute: dict[str, bytes]) -> None:
        pass


@pytest.fixture
def testcontext():
    return TestContext()


@pytest.fixture
def objectprovider(reference_image):
    attrs = TestAttributes()
    with BytesIO() as tmpfile:
        reference_image.save(tmpfile, format="JPEG", quality=85)
        content = tmpfile.getvalue()
    return ObjectProvider(obj_id="test", content=content, attributes=attrs)


@pytest.fixture
def reference_image():
    return Image.open(REFERENCE_IMAGE).convert("RGB")
