# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import io
import queue
import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence, cast

import numpy as np
import torch
import torchvision.transforms as transforms
from logzero import logger
from PIL import Image, ImageFile
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models

from ....classes import POSITIVE_CLASS
from ....detection import Detection
from ....proto.messages_pb2 import TestResults
from ...context.model_trainer_context import ModelContext
from ...core.model import ModelBase
from ...core.result_provider import ResultProvider
from ...core.utils import log_exceptions
from .config import FSLModelConfig

if TYPE_CHECKING:
    from ....hawkobject import HawkObject
    from ....objectid import ObjectId

torch.multiprocessing.set_sharing_strategy("file_system")
ImageFile.LOAD_TRUNCATED_IMAGES = True
device = "cuda" if torch.cuda.is_available() else "cpu"


class FSLModel(ModelBase):
    config: FSLModelConfig

    def __init__(
        self,
        config: FSLModelConfig,
        context: ModelContext,
        model_path: Path,
        version: int,
        *,
        train_examples: dict[str, int] | None = None,
        train_time: float = 0.0,
    ):
        logger.info(f"Loading FSL Model from {model_path}")
        assert model_path.is_file()

        test_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        super().__init__(
            config,
            context,
            model_path,
            version,
            train_examples,
            train_time,
        )

        self._test_transforms = test_transforms

        self._model = self.load_model(model_path)
        self._device = device
        self._model.to(self._device)
        self._model.eval()
        self._running = True

        support = Image.open(config.support_path).convert("RGB")
        self.support = self.get_embed(support)

    def get_embed(self, im: Image.Image) -> Sequence[float]:
        im = im.resize((224, 224))
        im_tensor = torch.unsqueeze(self._test_transforms(im), dim=0)
        with torch.no_grad():
            preds = self._model(im_tensor.to(device))
            return cast(Sequence[float], np.array([preds[0].cpu().numpy()]))

    @property
    def version(self) -> int:
        return self._version

    def preprocess(self, obj: HawkObject) -> Sequence[float]:
        assert obj.media_type.startswith("image/")

        image = Image.open(io.BytesIO(obj.content))
        if image.mode != "RGB":
            image = image.convert("RGB")

        return self.get_embed(image)

    def serialize(self) -> bytes:
        if self._model is None:
            return b""

        with io.BytesIO() as f:
            torch.save({"state_dict": self._model.state_dict()}, f)
            content = f.getvalue()
        return content

    def load_model(self, model_path: Path) -> torch.nn.Module:
        logger.info("Starting model load")
        model = models.resnet18().cuda()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        logger.info("Starting model complete")
        return model

    def get_predictions(self, inputs: torch.Tensor) -> Sequence[float]:
        # probability = []
        with torch.no_grad():
            similarity = cosine_similarity(self.support, inputs)
            if similarity.shape[-1] == 1:
                predictions = [float(similarity)]
            else:
                predictions = np.squeeze(similarity)
            return predictions

    @log_exceptions
    def _infer_results(self) -> None:
        logger.info("INFER RESULTS THREAD STARTED")

        requests = []
        timeout = 5
        next_infer = time.time() + timeout
        while self._running:
            try:
                request = self.request_queue.get(timeout=1)
                requests.append(request)
            except queue.Empty:
                continue

            if len(requests) == 0:
                continue

            if len(requests) < self.config.test_batch_size and time.time() < next_infer:
                continue

            results = self._process_batch(requests)
            for result in results:
                self.result_count += 1
                self.result_queue.put(result)

            requests = []
            next_infer = time.time() + timeout

    def infer(self, requests: Sequence[ObjectId]) -> Iterable[ResultProvider]:
        if not self._running or self._model is None:
            return []

        output = []
        for i in range(0, len(requests), self.config.test_batch_size):
            batch = []
            for object_id in requests[i : i + self.config.test_batch_size]:
                obj = self.context.retriever.get_ml_data(object_id)
                assert obj is not None
                batch.append((object_id, self.preprocess(obj)))
            results = self._process_batch(batch)
            for result in results:
                output.append(result)

        return output

    def _process_batch(
        self, batch: list[tuple[ObjectId, torch.Tensor]]
    ) -> Iterable[ResultProvider]:
        if self._model is None:
            if len(batch) > 0:
                # put request back in queue
                for req in batch:
                    self.request_queue.put(req)
            return

        with self._model_lock:
            tensors = np.stack([np.squeeze(f[1]) for f in batch])
            predictions = self.get_predictions(tensors)
            del tensors
            for i in range(len(batch)):
                score = predictions[i]
                bboxes = [Detection(class_name=POSITIVE_CLASS, confidence=score)]
                yield ResultProvider(batch[i][0], score, bboxes, self.version)

    def evaluate_model(self, test_path: Path) -> TestResults:
        raise Exception("ERROR: fsl.model.evaluate_model not implemented")

    def stop(self) -> None:
        logger.info(f"Stopping model of version {self.version}")
        with self._model_lock:
            self._running = False
            self._model = None
