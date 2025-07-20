# Copyright (c) 2025 Carnegie Mellon University
# SPDX-License-Identifier: MIT

import base64
import contextlib
from pathlib import Path

import pytest

from hawk.plugins import get_plugin_entrypoint

REFERENCE_IMAGE_DATA = (
    Path(__file__)
    .parent.parent.joinpath("assets", "grace_hopper_517x606.jpg")
    .read_bytes()
)

CONFIGS_MODEL = {
    "activity": {},
    "dnn_classifier": {},
    "dnn_classifier_radar": {},
    "fsl": {"support_data": base64.b64encode(REFERENCE_IMAGE_DATA).decode("utf-8")},
    "yolo": {},
    "yolo_radar": {},
}


class DummyMission:
    model_dir = Path("/tmp")


@pytest.mark.home
@pytest.mark.parametrize("model", CONFIGS_MODEL.keys())
def test_validate_trainer_config(model):
    with contextlib.suppress(ImportError):
        plugin_cls = get_plugin_entrypoint("model", model)
        plugin_cls.scrub_config(CONFIGS_MODEL[model])


@pytest.mark.scout
@pytest.mark.parametrize("model", CONFIGS_MODEL.keys())
def test_load_trainer_plugin(model):
    plugin = get_plugin_entrypoint("model", model)
    with contextlib.suppress(FileNotFoundError):
        plugin.from_config(CONFIGS_MODEL[model], context=DummyMission())
