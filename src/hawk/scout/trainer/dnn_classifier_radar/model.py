# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import io
import json
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, cast

import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms as transforms
from logzero import logger
from PIL import Image, ImageFile
from skimage.measure import label, regionprops
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


class DNNClassifierModelRadar(ModelBase):
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
        args["input_size"] = int(args.get("input_size", 224))
        radar_normalize = transforms.Normalize(
            mean=[0.111, 0.110, 0.111], std=[0.052, 0.050, 0.052]
        )
        test_transforms = transforms.Compose(
            [
                transforms.Pad(padding=(80, 0), fill=0, padding_mode="constant"),
                transforms.Resize(args["input_size"]),
                transforms.ToTensor(),
                radar_normalize,
            ]
        )

        args["test_batch_size"] = args.get("test_batch_size", 64)
        args["version"] = version
        args["arch"] = args.get("arch", "resnet50")
        args["train_examples"] = args.get("train_examples", {"1": 0, "0": 0})
        args["mode"] = mode
        args["pick_patches"] = args.get("pick_patches", False)
        self.args = args

        super().__init__(self.args, model_path, context)
        assert self.context is not None

        self._arch = args["arch"]
        self._train_examples = args["train_examples"]
        self._test_transforms = test_transforms
        self._batch_size = int(args["test_batch_size"])
        self.pick_patches = args["pick_patches"]

        self._model: torch.nn.Module = self.load_model(model_path)
        self._device = torch.device("cpu")
        self._model.to(self._device)
        self._model.eval()
        self._running = True

    @property
    def version(self) -> int:
        return self._version

    def preprocess(
        self, request: ObjectProvider
    ) -> tuple[ObjectProvider, torch.Tensor]:
        try:
            array = cast(npt.NDArray[Any], request.content)
            array = (array - np.min(array)) / (np.max(array) - np.min(array))
            image = Image.fromarray((array * 255).astype(np.uint8))
        except Exception:
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
        model = self.initialize_model(self._arch)
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

                if len(requests) < self._batch_size:
                    continue
            except queue.Empty:
                if len(requests) == 0 or time.time() < next_infer:
                    continue

            start_infer = time.time()
            results = self._process_batch(requests)
            logger.info(
                f"Process batch took {time.time()-start_infer}s for {len(requests)}"
            )
            for result in results:
                self.result_count += 1
                self.result_queue.put(result)

            requests = []
            # next_infer = start_infer + timeout
            next_infer = time.time() + timeout

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
        callback_fn: Callable[[int, list[int], list[Sequence[float]]], TestResults],
    ) -> TestResults:
        dataset = datasets.ImageFolder(str(directory), transform=self._test_transforms)
        data_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
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
            image_list, transform=self._test_transforms, label_list=label_list
        )
        data_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
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
        elif test_path.is_file():
            logger.info("Evaluating model")
            return self.infer_path(test_path, self.calculate_performance)
        else:
            raise Exception(f"ERROR: {test_path} does not exist")

    def _process_batch(
        self, batch: list[tuple[ObjectProvider, torch.Tensor]]
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
            if self.pick_patches:
                predictions, boxes = self.patch_processing(batch)
                # logger.info(f"Num pred:{len(predictions)}, Num boxes:{len(boxes)}")
            else:
                tensors = torch.stack([f[1] for f in batch])
                predictions = self.get_predictions(tensors)

            for i in range(len(batch)):
                score = predictions[i]
                box = [list(coord) for coord in boxes[i]]
                result_object = batch[i][0]
                if self._mode == "oracle":
                    num_classes = len(self.context.class_manager.classes)
                    cls = int(result_object.id.split("/", 2)[1])
                    score = [0.0] * num_classes
                    score[cls] = 1.0
                score_dict = {
                    label: float(score)
                    for label, score in zip(self.context.class_manager.classes, score)
                }
                result_object.attributes.add(
                    {"scores": json.dumps(score_dict).encode()}
                )
                result_object.attributes.add({"boxes": str.encode(str(box))})
                # add another attribute containing the estimated bounding boxes
                # should be a list of cls, x,y,w,h ground truth bounding boxes
                # will be added at home
                results.append(
                    ResultProvider(
                        result_object, sum(score[1:]), self.version  # , box)
                    )  ## score for priority queue is sum of all positive classes
                )
        return results

    def stop(self) -> None:
        logger.info(f"Stopping model of version {self.version}")
        with self._model_lock:
            self._running = False
            self._model = None

    def patch_processing(
        self, img_batch: list[tuple[ObjectProvider, torch.Tensor]]
    ) -> tuple[Sequence[Sequence[float]], list[list[tuple[int, int, int, int]]]]:
        num_image = 0
        tensors_for_predict = []
        crop_assign = []
        boxes_list: list[list[tuple[int, int, int, int]]] = []
        ## loop through batch and determine which samples have potential instances
        for obj, tensor in img_batch:
            # select the patches and crop
            cropped_images, coords = self.select_patches(obj.content)
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
                    [i for i in range(num_image, num_image + num_crops)]
                )  # not the list indices where these crops will be.
                boxes_list.append(coords)
                num_image += num_crops

        input_tensors = torch.stack([tens for tens in tensors_for_predict])
        # predictions include all cropped samples and those that are negs.
        predictions_tiles = self.get_predictions(input_tensors)
        predictions = []
        for crops in crop_assign:
            predictions_per_sample = np.array(
                predictions_tiles[crops[0] : crops[0] + len(crops)]
            )
            # across all patches for a given sample we combine the scores as folllows
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
        # in order to call this function, must check if self.args['pick_patches']
        # needs to return list of floats, which are the scores for the given
        # class for each sample.
        # logger.info(f"Predictions: {predictions}, box list: {boxes_list}")
        return predictions, boxes_list

    def select_patches(
        self, source_img: npt.NDArray[np.uint8]
    ) -> tuple[list[Image.Image], list[tuple[int, int, int, int]]]:
        # print(source_img.shape)
        threshold = 4.56 * 1.46  # Global median of negatives mult by constant factor
        binary_img = source_img.max(axis=2).transpose() > threshold
        binary_img[binary_img != 0] = 1
        labeled_binary_img = label(binary_img, connectivity=2)
        regions = regionprops(labeled_binary_img)

        condition = (
            lambda region: (
                (31 <= region.centroid[1] <= 33)
                and (region.bbox[3] - region.bbox[1] < 3)
            )
            or region.area < 4
        )
        regions = [region for region in regions if not condition(region)]  # type:ignore

        if not len(regions):
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
                left_border : right_border + 1, top_border : bottom_border + 1, :
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
