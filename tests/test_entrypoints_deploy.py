# Copyright (c) 2023 Carnegie Mellon University
# SPDX-License-Identifier: MIT

from hawk.deploy.__main__ import main as _deploy_main  # noqa: F401


def test_entrypoints_deploy():
    """The real test was if we could import the various entrypoints"""
    assert True
