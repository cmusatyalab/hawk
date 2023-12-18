# Copyright (c) 2023 Carnegie Mellon University
# SPDX-License-Identifier: MIT

from hawk.home.home_flutter import app as _flutter_app  # noqa: F401
from hawk.home.home_main import main as _home_main  # noqa: F401
from hawk.home.result_stream_new import main as _result_main  # noqa: F401
from hawk.scout.server_main import main as _scout_main  # noqa: F401
from hawk.scout.trainer.dnn_classifier.train_model import (  # noqa: F401
    main as _dnn_train_main,
)
from hawk.scout.trainer.fsl.augment import main as _fsl_augment_main  # noqa: F401

# the following depends on CUDA
# from hawk.scout.trainer.yolo.yolov5.train import main as _yolo_train_main


def test_dummy():
    """The real test was if we could import the various entrypoints"""
    assert True
