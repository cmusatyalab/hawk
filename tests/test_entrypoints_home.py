# Copyright (c) 2023 Carnegie Mellon University
# SPDX-License-Identifier: MIT

from hawk.home.home_flutter import app as _flutter_app  # noqa: F401
from hawk.home.home_main import main as _home_main  # noqa: F401
from hawk.home.result_stream_new import main as _result_main  # noqa: F401


def test_entrypoints_home():
    """The real test was if we could import the various entrypoints"""
    assert True
