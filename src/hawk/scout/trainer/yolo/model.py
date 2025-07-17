# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import gc
import io
import multiprocessing as mp
import os
import queue
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, cast

import numpy.typing as npt
import torch
import torchvision.transforms as transforms
from logzero import logger
from PIL import Image, ImageFile
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from torchvision import datasets

from ....classes import ClassLabel
from ....detection import Detection
from ....proto.messages_pb2 import TestResults
from ...context.model_trainer_context import ModelContext
from ...core.model import ModelBase
from ...core.result_provider import ResultProvider
from ...core.utils import log_exceptions

if TYPE_CHECKING:
    from ....hawkobject import HawkObject
    from ....objectid import ObjectId

torch.multiprocessing.set_sharing_strategy("file_system")
ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLOModel(ModelBase):
    def __init__(
        self,
        args: dict[str, Any],
        model_path: Path,
        version: int,
        mode: str,
        context: ModelContext,
    ):
        logger.info(f"Loading DNN Model from {model_path}")
        assert model_path.is_file()
        # args = dict(args)
        args["input_size"] = args.get("input_size", 480)
        test_transforms = transforms.Compose(
            [
                transforms.Resize(args["input_size"] + 32),
                transforms.CenterCrop(args["input_size"]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        args["test_batch_size"] = args.get("test_batch_size", 32)
        args["version"] = version
        args["train_examples"] = args.get("train_examples", {"1": 0, "0": 0})
        args["mode"] = mode
        self.args = args
        self.yolo_repo = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "yolov5"
        )

        super().__init__(self.args, model_path, context)

        self._test_transforms = test_transforms
        self._train_examples = args["train_examples"]
        self._batch_size = args["test_batch_size"]

        model = self.load_model(model_path)
        self._device = torch.device("cuda")
        self._model: torch.nn.Module | None = model.to(self._device)
        self._model.eval()
        self._running = True

    @property
    def version(self) -> int:
        return self._version

    def preprocess(self, obj: HawkObject) -> torch.Tensor:
        assert obj.media_type.startswith("image/")

        image = Image.open(io.BytesIO(obj.content))
        if image.mode != "RGB":
            image = image.convert("RGB")

        return self._test_transforms(image)

    def serialize(self) -> bytes:
        if self._model is None:
            return b""

        content = io.BytesIO()
        torch.save(
            {
                "state_dict": self._model.state_dict(),
            },
            content,
        )
        content.seek(0)

        return content.getvalue()

    def load_model(self, model_path: Path) -> torch.nn.Module:
        model = torch.hub.load(
            self.yolo_repo,
            "custom",
            path=str(model_path),
            source="local",
        )
        return model

    def get_predictions(
        self, inputs: torch.Tensor
    ) -> tuple[list[float], list[npt.NDArray]]:
        assert self._model is not None
        with torch.no_grad():
            output = self._model(inputs, detection=True).pred
            predictions = [out.cpu().numpy() for out in output]
            probability = [max(pred[:, 4]) if len(pred) else 0 for pred in predictions]
            logger.info(f"Returning probability:{probability}")
            return probability, predictions

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
                if len(requests) < self._batch_size:
                    continue
            except queue.Empty:
                if len(requests) == 0 or time.time() < next_infer:
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
        for i in range(0, len(requests), self._batch_size):
            batch = []
            for object_id in requests[i : i + self._batch_size]:
                obj = self.context.retriever.get_ml_data(object_id)
                assert obj is not None
                batch.append((object_id, self.preprocess(obj)))
            results = self._process_batch(batch)
            for result in results:
                output.append(result)

        return output

    def infer_dir(
        self,
        directory: Path,
        callback_fn: Callable[[Sequence[int], Sequence[float]], float],
    ) -> TestResults:
        dataset = datasets.ImageFolder(str(directory), transform=None)
        data_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=mp.cpu_count(),
        )

        targets = []
        predictions = []
        with torch.no_grad():
            for inputs, target in data_loader:
                prediction = self.get_predictions(inputs)
                del inputs
                for i in range(len(prediction)):
                    targets.append(target[i])
                    predictions.append(prediction[i])

        callback_fn(targets, predictions)
        raise Exception("ERROR: yolo.model.evaluate_model should return TestResults")

    def evaluate_model(self, test_path: Path) -> TestResults:
        def calculate_performance(
            y_true: Sequence[int], y_pred: Sequence[float]
        ) -> float:
            return cast(float, average_precision_score(y_true, y_pred, average=None))

        return self.infer_dir(test_path, calculate_performance)

    def _process_batch(
        self, batch: Sequence[tuple[ObjectId, torch.Tensor]]
    ) -> Iterable[ResultProvider]:
        if self._model is None:
            if len(batch) > 0:
                # put request back in queue
                for req in batch:
                    self.request_queue.put(req)
            return []

        results = []
        with self._model_lock:
            tensors = torch.stack([f[1] for f in batch]).to(self._device)
            prediction_scores, detections = self.get_predictions(tensors)
            del tensors

            assert self.context is not None

            for i in range(len(batch)):
                score = prediction_scores[i]
                detections_per_sample = detections[i]

                bboxes = [
                    Detection(
                        class_name=self.context.class_list[
                            ClassLabel(int(detections_per_sample[j, 5]) + 1)
                        ],
                        confidence=float(detections_per_sample[j, 4]),
                        center_x=float(detections_per_sample[j, 0] / 640),
                        center_y=float(detections_per_sample[j, 1] / 640),
                        width=float(detections_per_sample[j, 2] / 640),
                        height=float(detections_per_sample[j, 3] / 640),
                    )
                    for j in range(len(detections_per_sample))
                ]
                results.append(ResultProvider(batch[i][0], score, bboxes, self.version))
        return results

    def stop(self) -> None:
        logger.info(f"Stopping model of version {self.version}")
        with self._model_lock:
            self._running = False
            if self._model is not None:
                del self._model
                gc.collect()
                torch.cuda.empty_cache()
                self._model = None
