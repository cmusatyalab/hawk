# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import io
import multiprocessing as mp
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

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

torch.multiprocessing.set_sharing_strategy('file_system')
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DNNClassifierModel(ModelBase):

    def __init__(self,
                 args: Dict,
                 model_path: Path,
                 version: int,
                 mode: str,
                 context: ModelContext):

        logger.info(f"Loading DNN Model from {model_path}")
        assert model_path.is_file()
        args['input_size'] = args.get('input_size', 224)
        test_transforms = transforms.Compose([
            transforms.Resize(args['input_size'] + 32),
            transforms.CenterCrop(args['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        args['test_batch_size'] = args.get('test_batch_size', 64)
        args['version'] = version
        args['arch'] = args.get('arch', 'resnet50')
        args['train_examples'] = args.get('train_examples', {'1':0, '0':0})
        args['mode'] = mode
        self.args = args

        super().__init__(self.args, model_path, context)

        self._arch = args['arch']
        self._train_examples = args['train_examples']
        self._test_transforms = test_transforms
        self._batch_size = args['test_batch_size']

        self._model = self.load_model(model_path)
        self._device = torch.device('cpu')
        self._model.to(self._device)
        self._model.eval()
        self._running = True

    @property
    def version(self) -> int:
        return self._version

    def preprocess(self, request: ObjectProvider) -> Tuple[ObjectProvider, torch.Tensor]:
        try:
            image = Image.open(io.BytesIO(request.content))

            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise(e)

        return request, self._test_transforms(image)

    def serialize(self) -> bytes:
        if self._model is None:
            return None

        content = io.BytesIO()
        torch.save({
            'state_dict': self._model.state_dict(),
        }, content)
        content.seek(0)

        return content.getvalue()

    def load_model(self, model_path: Path):
        model = self.initialize_model(self._arch)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def initialize_model(self, arch, num_classes=2):
        model_ft = None
        model_ft = models.__dict__[arch](pretrained=True)

        if "resnet" in arch:
            """ Resnet
            """
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)

        elif "alexnet" in arch:
            """ Alexnet
            """
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = torch.nn.Linear(num_ftrs,num_classes)

        elif "vgg" in arch:
            """ VGG11_bn
            """
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = torch.nn.Linear(num_ftrs,num_classes)

        elif "squeezenet" in arch:
            """ Squeezenet
            """
            model_ft.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = num_classes

        elif "densenet" in arch:
            """ Densenet
            """
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = torch.nn.Linear(num_ftrs, num_classes)

        elif "inception" in arch:
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = torch.nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = torch.nn.Linear(num_ftrs,num_classes)

        elif "efficientnet" in arch:
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = torch.nn.Linear(num_ftrs,num_classes)

        else:
            logger.error("Invalid model name, exiting...")
            exit()

        return model_ft

    def get_predictions(self, inputs: torch.Tensor) -> List[float]:
        with torch.no_grad():
            inputs = inputs.to(self._device)
            output = self._model(inputs)

            probability = torch.softmax(output, dim=1)
            probability = probability.cpu().numpy()[:, 1]
            return probability

    @log_exceptions
    def _infer_results(self):
        logger.info("INFER RESULTS THREAD STARTED")

        requests = []
        timeout = 5
        prev_infer = time.time()
        while self._running:

            try:
                request = self.request_queue.get(block=False)
                requests.append(request)
            except Exception:
                # sleep when queue empty
                time.sleep(1)

            if not len(requests):
                continue

            if (len(requests) >=  self._batch_size or
                (time.time() - prev_infer) > timeout):
                prev_infer = time.time()
                results = self._process_batch(requests)
                for result in results:
                    self.result_count += 1
                    self.result_queue.put(result)
                requests = []
                #logger.info("Total results inferenced by model: {}".format(self.result_count))
                #logger.info("Request queue size: {}".format(self.request_queue.qsize()))

    def infer(self, requests: Iterable[ObjectProvider]) -> Iterable[ResultProvider]:
        if not self._running or self._model is None:
            return

        output = []
        for i in range(0, len(requests), self._batch_size):
            batch = []
            for request in requests[i:i+self._batch_size]:
                batch.append(self.preprocess(request))
            results = self._process_batch(batch)
            for result in results:
                output.append(result)

        return output

    def infer_dir(self, directory: Path, callback_fn: Callable[[int, float], None]) -> TestResults:
        dataset = datasets.ImageFolder(str(directory), transform=self._test_transforms)
        data_loader = DataLoader(dataset, batch_size=self._batch_size,
                                 shuffle=False, num_workers=mp.cpu_count())

        targets = []
        predictions = []
        with torch.no_grad():
            for inputs, target in data_loader:
                prediction = self.get_predictions(inputs)
                del inputs

                for i in range(len(prediction)):
                    targets.append(target[i])
                    predictions.append(prediction[i])

        return callback_fn(self.version, targets, predictions)

    def infer_path(self, test_file: Path, callback_fn: Callable[[int, float], None]) -> TestResults:

        image_list = []
        label_list = []
        with open(test_file) as f:
            contents = f.read().splitlines()
            for line in contents:
                path, label = line.split()
                image_list.append(path)
                label_list.append(int(label))

        dataset = ImageFromList(image_list, transform=self._test_transforms,
                                label_list=label_list)
        data_loader = DataLoader(dataset, batch_size=self._batch_size,
                                 shuffle=False, num_workers=mp.cpu_count())

        targets = []
        predictions = []
        with torch.no_grad():
            for inputs, target in data_loader:
                prediction = self.get_predictions(inputs)
                del inputs

                for i in range(len(prediction)):
                    targets.append(target[i])
                    predictions.append(prediction[i])

        return callback_fn(self.version, targets, predictions)

    def evaluate_model(self, test_path: Path) -> None:
        # call infer_dir
        self._device = torch.device('cuda')
        self._model.to(self._device)
        self._model.eval()

        if test_path.is_dir():
            return self.infer_dir(test_path, self.calculate_performance)
        elif test_path.is_file():
            logger.info("Evaluating model")
            return self.infer_path(test_path, self.calculate_performance)
        else:
            raise Exception(f'ERROR: {test_path} does not exist')

    def _process_batch(self, batch: List[Tuple[ObjectProvider, torch.Tensor]]) -> Iterable[ResultProvider]:
        if self._model is None:
            if len(batch) > 0:
                # put request back in queue
                for req in batch:
                    self.request_queue.put(req)
            return

        with self._model_lock:
            tensors = torch.stack([f[1] for f in batch])
            predictions = self.get_predictions(tensors)
            del tensors
            for i in range(len(batch)):
                score = predictions[i]
                result_object = batch[i][0]
                if self._mode == "oracle":
                    if '/0/' in result_object.id:
                        score = 0
                    else:
                        score = 1
                result_object.attributes.add({'score': str.encode(str(score))})
                yield ResultProvider(result_object, score,  self.version)

    def stop(self):
        logger.info(f"Stopping model of version {self.version}")
        with self._model_lock:
            self._running = False
            self._model = None
