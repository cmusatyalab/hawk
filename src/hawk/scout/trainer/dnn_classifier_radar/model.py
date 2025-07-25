# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import io
import multiprocessing as mp
import queue
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

import numpy as np
import numpy.typing as npt
import torch
from logzero import logger
from PIL import Image, ImageFile
from skimage.measure import label, regionprops
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from ....detection import Detection
from ...core.model import ModelBase
from ...core.result_provider import ResultProvider
from ...core.utils import ImageFromList, log_exceptions

if TYPE_CHECKING:
    from ....hawkobject import HawkObject
    from ....objectid import ObjectId
    from ....proto.messages_pb2 import TestResults
    from ...context.model_trainer_context import ModelContext
    from .config import DNNRadarModelConfig

torch.multiprocessing.set_sharing_strategy("file_system")
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DNNClassifierModelRadar(ModelBase):
    config: DNNRadarModelConfig

    def __init__(
        self,
        config: DNNRadarModelConfig,
        context: ModelContext,
        model_path: Path,
        version: int,
        *,
        train_examples: dict[str, int] | None = None,
        train_time: float = 0.0,
    ) -> None:
        logger.info(f"Loading DNN Model from {model_path}")
        assert model_path.is_file()
        radar_normalize = transforms.Normalize(
            mean=[0.111, 0.110, 0.111],
            std=[0.052, 0.050, 0.052],
        )
        test_transforms = transforms.Compose(
            [
                transforms.Pad(padding=(80, 0), fill=0, padding_mode="constant"),
                transforms.Resize(config.input_size),
                transforms.ToTensor(),
                radar_normalize,
            ],
        )

        super().__init__(
            config,
            context,
            model_path,
            version,
            train_examples,
            train_time,
        )
        assert self.context is not None

        self._test_transforms = test_transforms

        self._model: torch.nn.Module = self.load_model(model_path)
        self._device = torch.device("cpu")
        self._model.to(self._device)
        self._model.eval()
        self._running = True

    @property
    def version(self) -> int:
        return self._version

    def preprocess(self, obj: HawkObject) -> torch.Tensor:
        if obj.media_type in ("x-array/numpy", "x-array/numpyz"):
            array = np.load(io.BytesIO(obj.content))
            array = (array - np.min(array)) / (np.max(array) - np.min(array))
            image = Image.fromarray((array * 255).astype(np.uint8))
        else:
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
        return content.getvalue()

    def load_model(self, model_path: Path) -> torch.nn.Module:
        model = self.initialize_model(self.config.arch)
        logger.info(f"Loading new model from path..,{model_path}")
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
                512,
                num_classes,
                kernel_size=(1, 1),
                stride=(1, 1),
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
            # Handle the auxiliary net
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
            sys.exit()

        return model_ft

    def get_predictions(self, inputs: torch.Tensor) -> Sequence[Sequence[float]]:
        with torch.no_grad():
            inputs = inputs.to(self._device)
            output = self._model(inputs)

            probability: torch.Tensor = torch.softmax(output, dim=1)
            predictions: Sequence[Sequence[float]] = probability.cpu().numpy()
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

                if len(requests) < self.config.test_batch_size and self._running:
                    continue
            except queue.Empty:
                if (len(requests) == 0 or time.time() < next_infer) and self._running:
                    continue

            start_infer = time.time()
            results = self._process_batch(requests)
            logger.info(
                f"Process batch took {time.time() - start_infer}s for {len(requests)}",
            )
            for result in results:
                self.result_count += 1
                self.result_queue.put(result)

            requests = []
            # next_infer = start_infer + timeout
            next_infer = time.time() + timeout
            logger.info(f"Total results inferenced by model: {self.result_count}")

    def infer(self, requests: Sequence[ObjectId]) -> Iterable[ResultProvider]:
        if not self._running or self._model is None:
            return

        for i in range(0, len(requests), self.config.test_batch_size):
            batch = []
            for object_id in requests[i : i + self.config.test_batch_size]:
                obj = self.context.retriever.get_ml_data(object_id)
                assert obj is not None
                batch.append((object_id, self.preprocess(obj)))
            results = self._process_batch(batch)
            yield from results

    def infer_dir(
        self,
        directory: Path,
        callback_fn: Callable[[int, list[int], list[Sequence[float]]], TestResults],
    ) -> TestResults:
        dataset = datasets.ImageFolder(str(directory), transform=self._test_transforms)
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=mp.cpu_count(),
        )

        targets: list[int] = []
        predictions: list[Sequence[float]] = []
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
        callback_fn: Callable[[int, list[int], list[Sequence[float]]], TestResults],
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
            image_list,
            transform=self._test_transforms,
            label_list=label_list,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=mp.cpu_count(),
        )

        targets: list[int] = []
        predictions: list[Sequence[float]] = []
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
        if test_path.is_file():
            logger.info("Evaluating model")
            return self.infer_path(test_path, self.calculate_performance)
        msg = f"ERROR: {test_path} does not exist"
        raise Exception(msg)

    def _process_batch(
        self,
        batch: list[tuple[ObjectId, torch.Tensor]],
    ) -> Iterable[ResultProvider]:
        assert self.context is not None
        if self._model is None:
            if len(batch) > 0:
                # put request back in queue
                for req in batch:
                    self.request_queue.put(req)
            return []

        results = []
        # need additional function call here to crop out the high prob areas of RD map
        # if we crop and add samples to the batch here, then the batch size will be > 64

        with self._model_lock:
            if self.config.pick_patches:
                predictions, boxes = self.patch_processing(batch)
                # logger.info(f"Num pred:{len(predictions)}, Num boxes:{len(boxes)}")
                logger.info("Pick patches is True")
            else:
                tensors = torch.stack([f[1] for f in batch])
                predictions = self.get_predictions(tensors)

            for i in range(len(batch)):
                score = predictions[i]
                result_object = batch[i][0]

                # if pick_patches is true we should include x y w h
                # this doesn't work because boxes is a list[list[tuple[..]]]?
                # if self.config.pick_patches:
                #    l, r, t, b = boxes[i]
                #    x = (r + l) / (2 * 63)
                #    y = (b + t) / (2 * 255)
                #    w = (r - l) / 63
                #    h = (b - t) / 255
                # else:
                x, y, w, h = 0.5, 0.5, 1.0, 1.0

                bboxes = [
                    Detection(
                        class_name=class_name,
                        confidence=float(score),
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                    )
                    for class_name, score in zip(self.context.class_list, score)
                ]
                # score for priority queue is sum of all positive classes
                results.append(
                    ResultProvider(
                        result_object,
                        sum(score[1:]),
                        bboxes,
                        self.version,
                        None,
                    ),
                )
        return results

    def stop(self) -> None:
        logger.info(f"Stopping model of version {self.version}")
        with self._model_lock:
            self._running = False
            self._model = None

    def patch_processing(
        self,
        img_batch: list[tuple[ObjectId, torch.Tensor]],
    ) -> tuple[Sequence[Sequence[float]], list[list[tuple[int, int, int, int]]]]:
        num_image = 0
        tensors_for_predict = []
        crop_assign = []
        boxes_list: list[list[tuple[int, int, int, int]]] = []
        ## loop through batch and determine which samples have potential instances
        for object_id, tensor in img_batch:
            obj = self.context.retriever.get_ml_data(object_id)
            assert obj is not None
            assert obj.media_type in (
                "x-array/numpy",
                "x-array/numpyz",
            )

            # select the patches and crop
            array = np.load(io.BytesIO(obj.content))
            cropped_images, coords = self.select_patches(array)
            num_crops = len(cropped_images)

            if not num_crops:  # no potential objects in sample
                tensors_for_predict.append(tensor)
                crop_assign.append([num_image])
                boxes_list.append([])
                num_image += 1
            else:  # at least 1 instance in the source image
                for crop in cropped_images:
                    tensors_for_predict.append(crop)
                crop_assign.append(
                    list(range(num_image, num_image + num_crops)),
                )  # not the list indices where these crops will be.
                boxes_list.append(coords)
                num_image += num_crops

        input_tensors = torch.stack(list(tensors_for_predict))
        # predictions include all cropped samples and those that are negs.
        predictions_tiles = self.get_predictions(input_tensors)
        predictions = []
        for crops in crop_assign:
            predictions_per_sample = np.array(
                predictions_tiles[crops[0] : crops[0] + len(crops)],
            )
            # across all patches for a given sample we combine the scores as follows
            # to make sure negative results don't diminish positive detections.
            # collect highest confidence scores for all positive classes
            # take lowest confidence for negative class
            # renormalize to make sure everything still adds up to 1
            min_negative = predictions_per_sample[:, 0:1].min(axis=0)
            max_positive = predictions_per_sample[:, 1:].max(axis=0)
            new_predictions = np.hstack([min_negative, max_positive])
            new_predictions /= np.sum(new_predictions)
            predictions.append(list(new_predictions))

        # find the set of n patches that are most likely to contain an object.
        # select a few additional patches of negative space
        # return a list of patches, perhaps do a yield
        # in order to call this function, must check if self.config.pick_patches
        # needs to return list of floats, which are the scores for the given
        # class for each sample.
        # logger.info(f"Predictions: {predictions}, box list: {boxes_list}")
        return predictions, boxes_list

    def select_patches(
        self,
        source_img: npt.NDArray[np.uint8],
    ) -> tuple[list[Image.Image], list[tuple[int, int, int, int]]]:
        # print(source_img.shape)
        threshold = 4.56 * 1.46  # Global median of negatives mult by constant factor
        binary_img = source_img.max(axis=2).transpose() > threshold
        binary_img[binary_img != 0] = 1

        # skimage.measure seems to be untyped
        labeled_binary_img = label(binary_img, connectivity=2)  # type: ignore[no-untyped-call]
        regions = regionprops(labeled_binary_img)  # type: ignore[no-untyped-call]

        class _RegionProperties:
            area: float
            bbox: tuple[int, int, int, int]
            centroid: tuple[int, int]

        def condition(region: _RegionProperties) -> bool:
            return (
                (31 <= region.centroid[1] <= 33)
                and (region.bbox[3] - region.bbox[1] < 3)
            ) or region.area < 4

        regions = [region for region in regions if not condition(region)]

        if not regions:
            return [], []
        box_distance_xthresh, box_distance_ythresh = 10, 10
        merged_regions: list[tuple[int, int, int, int]] = []
        for region in regions:
            bbox = region.bbox
            width = bbox[3] - bbox[1]
            height = bbox[2] - bbox[0]
            ### merge smaller regions to larger regions
            is_adjacent = False
            for merged_region in merged_regions:
                cond_1 = region.bbox[0] - merged_region[2] <= box_distance_ythresh
                cond_2 = merged_region[0] - region.bbox[2] <= box_distance_ythresh
                cond_3 = region.bbox[1] - merged_region[3] <= box_distance_xthresh
                cond_4 = merged_region[1] - region.bbox[3] <= box_distance_xthresh
                if cond_1 and cond_2 and cond_3 and cond_4:
                    minr = min(region.bbox[0], merged_region[0])
                    minc = min(region.bbox[1], merged_region[1])
                    maxr = max(region.bbox[2], merged_region[2])
                    maxc = max(region.bbox[3], merged_region[3])
                    merged_regions.append((minr, minc, maxr, maxc))
                    merged_regions.remove(merged_region)
                    is_adjacent = True
                    break
            if not is_adjacent:
                merged_regions.append(region.bbox)

        padded_crops = []
        coords_list = []
        for bbox in merged_regions:
            width, height = bbox[3] - bbox[1], bbox[2] - bbox[0]
            centerx = int(width / 2 + bbox[1])
            centery = int(height / 2 + bbox[0])
            left_border = max(0, centerx - int(max(width / 2, 25)))
            right_border = min(63, centerx + int(max(width / 2, 25)))
            top_border = max(0, centery - int(max(height / 2, 37)))
            bottom_border = min(255, centery + int(max(height / 2, 37)))
            norm_source_img = (source_img - np.min(source_img)) / (
                np.max(source_img) - np.min(source_img)
            )  # normalize
            cropped_image = norm_source_img[
                left_border : right_border + 1,
                top_border : bottom_border + 1,
                :,
            ]
            padded_image = np.pad(
                cropped_image,
                (
                    (left_border, 63 - right_border),
                    (top_border, 255 - bottom_border),
                    (0, 0),
                ),
                mode="constant",
            )
            padded_pilimage = Image.fromarray((padded_image * 255).astype(np.uint8))
            padded_pilimage = self._test_transforms(padded_pilimage)
            padded_crops.append(padded_pilimage)
            coords_list.append((left_border, right_border, top_border, bottom_border))
        return padded_crops, coords_list
