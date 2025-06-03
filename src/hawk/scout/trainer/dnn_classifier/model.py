# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import io
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Sequence, Tuple

import torch
from logzero import logger
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets

from ....classes import class_label_to_int
from ....proto.messages_pb2 import TestResults
from ...context.model_trainer_context import ModelContext
from ...core.model import ModelBase
from ...core.result_provider import BoundingBox, ResultProvider
from ...core.utils import ImageFromList, log_exceptions
from .training_state import TrainingState

if TYPE_CHECKING:
    from ....hawkobject import HawkObject
    from ....objectid import ObjectId

torch.multiprocessing.set_sharing_strategy("file_system")
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DNNClassifierModel(ModelBase):
    def __init__(
        self,
        args: Dict[str, Any],
        model_path: Path,
        version: int,
        mode: str,
        context: ModelContext,
    ):
        logger.info(f"Loading DNN Model from {model_path}")
        assert model_path.is_file()
        args["input_size"] = int(args.get("input_size", 224))
        args["test_batch_size"] = args.get("test_batch_size", 64)
        args["version"] = version
        args["arch"] = args.get("arch", "resnet50")
        args["train_examples"] = args.get("train_examples", {"1": 0, "0": 0})
        args["mode"] = mode
        self.args = args

        super().__init__(self.args, model_path, context)
        assert self.context is not None

        self._arch = args["arch"]
        self._train_examples = args["train_examples"]
        self._batch_size = args["test_batch_size"]

        self._model, self._preprocess = self.load_model(model_path)
        self._device = torch.device("cpu")
        self._model.to(self._device)
        self._model.eval()
        self._running = True
        self.extract_feature_vector = False
        if self.context.novel_class_discovery or self.context.sub_class_discovery:
            self.extract_feature_vector = True

    @property
    def version(self) -> int:
        return self._version

    def preprocess(self, obj: HawkObject) -> torch.Tensor:
        assert obj.media_type.startswith("image/")

        image = Image.open(io.BytesIO(obj.content))
        if image.mode != "RGB":
            image = image.convert("RGB")

        return self._preprocess(image)

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
        return content.getvalue()

    def load_model(self, model_path: Path) -> torch.nn.Module:
        assert self.context is not None
        model, preprocess, _ = TrainingState.load_for_inference(model_path, self._arch)
        return model, preprocess

    def get_predictions(self, inputs: torch.Tensor) -> Sequence[Sequence[float]]:
        with torch.no_grad():
            if self.extract_feature_vector:
                self._model.avgpool.register_forward_hook(self.forward_hook)
                # avgpool module is based on the resnet architecture, will need
                # to add additional support for appropriate layer to extract
                # feature vector from other models.
            inputs = inputs.to(self._device)
            output = self._model(inputs)
            probability: torch.Tensor = torch.softmax(output, dim=1)
            predictions: Sequence[Sequence[float]] = (
                probability.cpu().numpy()
            )  # [:, 1]  ## changed this to output multi-class vector
            return predictions

    @log_exceptions
    def _infer_results(self) -> None:
        logger.info("INFER RESULTS THREAD STARTED")

        requests = []
        timeout = 5
        next_infer = time.time() + timeout
        while self._running:  ## or len(requests) > 0:
            try:
                request = self.request_queue.get(timeout=1)
                requests.append(request)

                if len(requests) < self._batch_size and self._running:
                    # what if len(requests) is 1 and self._running is False?
                    # Inference the last sample or put back into the queue?
                    # self.request_queue.put(request)
                    continue
            except queue.Empty:
                if (len(requests) == 0 or time.time() < next_infer) and self._running:
                    # logger.info(
                    #     "May be during model transition,"
                    #     f" we lose samples: {len(requests)},"
                    #     f" delay too short: {time.time()} < {next_infer},"
                    #     f" self._running: {self._running}"
                    # )
                    # or we could do a put back into the request_queue
                    continue

            start_infer = time.time()
            # logger.info(f"\nFeeding {len(requests)} to inference...\n")
            results = self._process_batch(requests)
            logger.info(
                f"Process batch took {time.time()-start_infer}s for {len(requests)}"
            )
            for result in results:
                self.result_count += 1
                self.result_queue.put(result)
                # this is a possible place to add results to a queue for clustering

            requests = []
            # next_infer = start_infer + timeout
            next_infer = time.time() + timeout
            # logger.info(f"Total results inferenced by model: {self.result_count}")
            # logger.info(f"Request queue size: {self.request_queue.qsize()}")

    def infer(self, requests: Sequence[ObjectId]) -> Iterable[ResultProvider]:
        if not self._running or self._model is None:
            return

        for i in range(0, len(requests), self._batch_size):
            batch = []
            for object_id in requests[i : i + self._batch_size]:
                obj = self.context.retriever.get_ml_data(object_id)
                assert obj is not None
                batch.append((object_id, self.preprocess(obj)))
            results = self._process_batch(batch)
            yield from results

    def infer_dir(
        self,
        directory: Path,
        callback_fn: Callable[[int, List[int], List[Sequence[float]]], TestResults],
    ) -> TestResults:
        dataset = datasets.ImageFolder(str(directory), transform=self._preprocess)
        data_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=mp.cpu_count(),
        )

        targets: List[int] = []
        predictions: List[Sequence[float]] = []
        with torch.no_grad():
            for inputs, target in data_loader:
                prediction = self.get_predictions(inputs)
                del inputs

                for i in range(len(prediction)):
                    targets.append(target[i])
                    predictions.append(prediction[i])

        return callback_fn(self.version, targets, predictions)

    def infer_path(
        self,
        test_file: Path,
        callback_fn: Callable[[int, List[int], List[Sequence[float]]], TestResults],
    ) -> TestResults:
        image_list = []
        label_list = []
        with open(test_file) as f:
            contents = f.read().splitlines()
            for line in contents:
                path, label = line.split()
                image_list.append(Path(path))
                label_list.append(int(label))

        dataset = ImageFromList(
            image_list, transform=self._preprocess, label_list=label_list
        )
        data_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=mp.cpu_count(),
        )

        targets: List[int] = []
        predictions: List[Sequence[float]] = []
        with torch.no_grad():
            for inputs, target in data_loader:
                prediction = self.get_predictions(inputs)
                del inputs

                for i in range(len(prediction)):
                    targets.append(target[i])
                    predictions.append(prediction[i])

        return callback_fn(self.version, targets, predictions)

    def evaluate_model(self, test_path: Path) -> TestResults:
        # call infer_dir
        self._device = torch.device("cuda")
        self._model.to(self._device)
        self._model.eval()

        if test_path.is_dir():
            return self.infer_dir(test_path, self.calculate_performance)
        elif test_path.is_file():
            logger.info("Evaluating model")
            return self.infer_path(test_path, self.calculate_performance)
        else:
            raise Exception(f"ERROR: {test_path} does not exist")

    def _process_batch(
        self, batch: List[Tuple[ObjectId, torch.Tensor]]
    ) -> Iterable[ResultProvider]:
        assert self.context is not None
        if self._model is None:
            if len(batch) > 0:
                # put request back in queue
                for req in batch:
                    self.request_queue.put(req)
            return []

        results = []
        with self._model_lock:
            tensors = torch.stack([f[1] for f in batch])
            predictions = self.get_predictions(tensors)
            del tensors
            for i in range(len(batch)):
                score = predictions[i]
                result_object = batch[i][0]

                if self.extract_feature_vector:
                    feature_vector = self.batch_feature_vectors[i]
                    with io.BytesIO() as fv_bytes:
                        torch.save(feature_vector, fv_bytes)
                        final_fv = fv_bytes.getvalue()
                else:
                    final_fv = None

                if self._mode == "oracle":
                    num_classes = len(self.context.class_list)

                    class_name = result_object._groundtruth()
                    class_label = self.context.class_list.index(class_name)

                    score = [0.0] * num_classes
                    score[class_label_to_int(class_label)] = 1.0

                bboxes: list[BoundingBox] = [
                    {
                        "class_name": class_name,
                        "confidence": float(score),
                    }
                    for class_name, score in zip(self.context.class_list, score)
                ]
                results.append(
                    ResultProvider(
                        result_object,
                        sum(score[1:]),
                        bboxes,
                        self.version,
                        final_fv,
                    )  ## score for priority queue is sum of all positive classes
                )
        return results

    def stop(self) -> None:
        logger.info(f"Stopping model of version {self.version}")
        with self._model_lock:
            self._running = False
            self._model = None

    def forward_hook(
        self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> None:
        self.batch_feature_vectors = output.detach().cpu()
        self.batch_feature_vectors = self.batch_feature_vectors.reshape(
            self.batch_feature_vectors.shape[0], -1
        )
