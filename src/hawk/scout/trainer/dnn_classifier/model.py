# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import io
import json
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import torch
import torchvision.transforms as transforms
from logzero import logger
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, models

from ....proto.messages_pb2 import TestResults
from ...context.model_trainer_context import ModelContext
from ...core.model import ModelBase
from ...core.object_provider import ObjectProvider
from ...core.result_provider import ResultProvider
from ...core.utils import ImageFromList, log_exceptions

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
        self._test_transforms = test_transforms
        self._batch_size = args["test_batch_size"]

        self._model: torch.nn.Module = self.load_model(model_path)
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

    def preprocess(
        self, request: ObjectProvider
    ) -> Tuple[ObjectProvider, torch.Tensor]:
        assert isinstance(request.content, bytes)
        image = Image.open(io.BytesIO(request.content))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return request, self._test_transforms(image)

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
        model = self.initialize_model(
            self._arch, num_classes=len(self.context.class_manager.classes)
        )
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def initialize_model(self, arch: str, num_classes: int = 2) -> torch.nn.Module:
        model_ft = models.__dict__[arch](pretrained=True)

        if "resnet" in arch:
            """Resnet"""
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)

        elif "alexnet" in arch:
            """Alexnet"""
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)

        elif "vgg" in arch:
            """VGG11_bn"""
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)

        elif "squeezenet" in arch:
            """Squeezenet"""
            model_ft.classifier[1] = torch.nn.Conv2d(
                512, num_classes, kernel_size=(1, 1), stride=(1, 1)
            )
            model_ft.num_classes = num_classes

        elif "densenet" in arch:
            """Densenet"""
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = torch.nn.Linear(num_ftrs, num_classes)

        elif "inception" in arch:
            """Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = torch.nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)

        elif "efficientnet" in arch:
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)

        else:
            logger.error("Invalid model name, exiting...")
            exit()

        return model_ft

    def get_predictions(self, inputs: torch.Tensor) -> Sequence[Sequence[float]]:
        with torch.no_grad():
            if self.extract_feature_vector:
                handle = self._model.avgpool.register_forward_hook(self.forward_hook)
                ## avgpool module is based on the resnet architecture, will need to add additional support for appropriate layer to extract feature vector from other models.
            inputs = inputs.to(self._device)
            output = self._model(inputs)
            probability: torch.Tensor = torch.softmax(output, dim=1)
            predictions: Sequence[
                Sequence[float]
            ] = (
                probability.cpu().numpy()
            )  # [:, 1]  ## changed this to output multi-class vector
            return predictions

    @log_exceptions
    def _infer_results(self) -> None:
        logger.info("INFER RESULTS THREAD STARTED")

        requests = []
        timeout = 5
        next_infer = time.time() + timeout
        while self._running: ## or len(requests) > 0:
            try:
                request = self.request_queue.get(timeout=1)
                requests.append(request)

                if len(requests) < self._batch_size and self._running: ## what if len(requests) is 1 and self._running is False?  Inference the last sample or put back into the inference queue: self.request_queue.put(request)
                    continue
            except queue.Empty:
                if (len(requests) == 0 or time.time() < next_infer) and self._running:
                    #logger.info(f"\nMay be during model transition, we lose samples: {len(requests)}, delay too short: {time.time()}, < {next_infer}, self._running: {self._running}") ## or we could do a put back into the request_queue
                    continue

            start_infer = time.time()
            #logger.info(f"\nFeeding {len(requests)} to inference...\n")
            results = self._process_batch(requests)
            logger.info(
                f"Process batch took {time.time()-start_infer}s for {len(requests)}"
            )
            for result in results:
                self.result_count += 1
                self.result_queue.put(result)
                ## this is a possible place to add results to another queue for clustering 

            requests = []
            # next_infer = start_infer + timeout
            next_infer = time.time() + timeout
            # logger.info(f"Total results inferenced by model: {self.result_count}")
            # logger.info(f"Request queue size: {self.request_queue.qsize()}")

    def infer(self, requests: Sequence[ObjectProvider]) -> Iterable[ResultProvider]:
        if not self._running or self._model is None:
            return

        for i in range(0, len(requests), self._batch_size):
            batch = []
            for request in requests[i : i + self._batch_size]:
                batch.append(self.preprocess(request))
            results = self._process_batch(batch)
            yield from results

    def infer_dir(
        self,
        directory: Path,
        callback_fn: Callable[[int, List[int], List[Sequence[float]]], TestResults],
    ) -> TestResults:
        dataset = datasets.ImageFolder(str(directory), transform=self._test_transforms)
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
            image_list, transform=self._test_transforms, label_list=label_list
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
        self, batch: List[Tuple[ObjectProvider, torch.Tensor]]
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
                feature_vector = self.batch_feature_vectors[i] if self.extract_feature_vector else None
                fv_bytes = io.BytesIO()
                torch.save(feature_vector, fv_bytes)
                final_fv = fv_bytes.getvalue() if feature_vector is not None else None
                if self._mode == "oracle":
                    num_classes = len(self.context.class_manager.classes)
                    cls = int(result_object.id.split("/", 2)[1])
                    score = [0.0] * num_classes
                    score[cls] = 1.0
                score_dict = {
                    label: float(score)
                    for label, score in zip(self.context.class_manager.classes, score)
                }
                detection_list = [
                    {
                     'cls_scores': score_dict
                    }
                ]
                result_object.attributes.add(
                    {"detections": json.dumps(detection_list).encode()}
                )
                results.append(
                    ResultProvider(
                        result_object, sum(score[1:]), self.version, final_fv
                    )  ## score for priority queue is sum of all positive classes
                )
        return results

    def stop(self) -> None:
        logger.info(f"Stopping model of version {self.version}")
        with self._model_lock:
            self._running = False
            self._model = None


    def forward_hook(self, module, input, output):
        self.batch_feature_vectors = output.detach().cpu()
        self.batch_feature_vectors = self.batch_feature_vectors.reshape(self.batch_feature_vectors.shape[0],-1)