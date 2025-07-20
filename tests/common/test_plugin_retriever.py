# Copyright (c) 2025 Carnegie Mellon University
# SPDX-License-Identifier: MIT

import contextlib

import pytest

from hawk.plugins import get_plugin_entrypoint

CONFIGS_RETRIEVER = {
    "frame": {"index_path": "/example/path/to/index"},
    #    "http": {"base_url": "http://localhost"},
    #    "k600": {"root": "/path/to/index"},
    "network": {
        "index_path": "/example/path/to/index",
        "server_host": "localhost",
        "server_port": 8000,
        "balance_mode": "locally_constant",
    },
    "random": {"index_path": "/example/path/to/index"},
    "tile": {"index_path": "/example/path/to/index"},
    "video": {"video_path": "/example/path/to/index"},
}

CONFIG_OVERRIDE = {"mission_id": "", "data_root": "/example"}


@pytest.mark.home
@pytest.mark.parametrize("retriever", CONFIGS_RETRIEVER.keys())
def test_validate_retriever_config(retriever):
    with contextlib.suppress(ImportError):
        plugin_cls = get_plugin_entrypoint("retriever", retriever)
        plugin_cls.scrub_config(
            dict(CONFIGS_RETRIEVER[retriever], **CONFIG_OVERRIDE),
            exclude=set(CONFIG_OVERRIDE),
        )


@pytest.mark.scout
@pytest.mark.parametrize("retriever", CONFIGS_RETRIEVER.keys())
def test_load_retriever_plugin(retriever):
    config = dict(CONFIGS_RETRIEVER[retriever], **CONFIG_OVERRIDE)
    plugin = get_plugin_entrypoint("retriever", retriever)
    with contextlib.suppress(FileNotFoundError):
        plugin.from_config(config)
