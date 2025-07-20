# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from hawk import Detection, ObjectId
from hawk.classes import POSITIVE_CLASS
from hawk.proto.messages_pb2 import TestResults
from hawk.scout.context.model_trainer_context import ModelContext
from hawk.scout.core.config import ModelConfig, ModelTrainerConfig
from hawk.scout.core.model import ModelBase
from hawk.scout.core.model_trainer import ModelTrainer
from hawk.scout.core.result_provider import ResultProvider


class ExampleModelConfig(ModelConfig):
    pass


class ExampleTrainerConfig(ExampleModelConfig, ModelTrainerConfig):
    pass


class ExampleModel(ModelBase):
    config: ExampleModelConfig

    def __init__(
        self,
        config: ExampleModelConfig,
        context: ModelContext,
        model_path: Path,
        version: int,
        *,
        train_examples: dict[str, int] | None = None,
        train_time: float = 0.0,
    ):
        super().__init__(
            config,
            context,
            model_path,
            version,
            train_examples=train_examples,
            train_time=train_time,
        )
        self.load_model(model_path)
        self._is_running = True

    def load_model(self, path: Path) -> None:
        pass

    def serialize(self) -> bytes:
        return b""

    def evaluate_model(self, test_path: Path) -> TestResults:
        return TestResults()

    def stop(self) -> None:
        self._is_running = False

    def infer(self, requests: Sequence[ObjectId]) -> Iterable[ResultProvider]:
        # Everything is awesome!
        for object_id in requests:
            yield ResultProvider(
                object_id, 1.0, [Detection(class_name=POSITIVE_CLASS)], self.version
            )


class ExampleTrainer(ModelTrainer):
    """Example trainer, just returns the same model."""

    config_class = ExampleTrainerConfig
    config: ExampleTrainerConfig

    def __init__(self, config: ExampleTrainerConfig, context: ModelContext):
        super().__init__(config, context)

    def load_model(
        self, path: Path | None = None, content: bytes = b"", version: int = -1
    ) -> ExampleModel:
        new_version = self.get_new_version()

        if path is None or not path.is_file():
            assert len(content)
            path = self.context.model_path(new_version)
            path.write_bytes(content)

        self.prev_path = path
        version = self.get_version()
        return ExampleModel(self.config, self.context, path, version)

    def train_model(self, train_dir: Path) -> ExampleModel:
        new_version = self.get_new_version()
        return self.load_model(self.prev_path, version=new_version)
