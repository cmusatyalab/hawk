# Copyright (c) 2023-2024 Carnegie Mellon University
# SPDX-License-Identifier: MIT

import pytest

from hawk.deploy.__main__ import main as _deploy_main  # noqa: F401
from hawk.home.home_flutter import app as _flutter_app  # noqa: F401
from hawk.home.home_main import main as _home_main  # noqa: F401
from hawk.home.result_stream_new import main as _result_main  # noqa: F401


@pytest.mark.home
def test_entrypoints_home():
    """The real test was if we could import the various entrypoints"""
    assert True
