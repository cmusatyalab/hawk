# Copyright (c) 2025 Carnegie Mellon University
# SPDX-License-Identifier: MIT

import contextlib

import pytest

from hawk.scout.retrieval.loader import load_retriever

CONFIGS = {
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


@pytest.mark.home
@pytest.mark.parametrize("retriever", CONFIGS.keys())
def test_validate_retriever_config(retriever):
    config = dict(CONFIGS[retriever], mission_id="", data_root="/example")
    with contextlib.suppress(ImportError):
        retriever_cls = load_retriever(retriever)
        retriever_cls.validate_config(config)


@pytest.mark.scout
@pytest.mark.parametrize("retriever", CONFIGS.keys())
def test_load_retriever(retriever):
    config = dict(CONFIGS[retriever], mission_id="", data_root="/example")
    retriever_cls = load_retriever(retriever)
    with contextlib.suppress(FileNotFoundError):
        retriever_cls.from_config(config)
