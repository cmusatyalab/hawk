# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import io
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from logzero import logger
from PIL import Image, ImageFile
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models

from ...context.model_trainer_context import ModelContext
from ...core.model import ModelBase
from ...core.object_provider import ObjectProvider
from ...core.result_provider import ResultProvider
from ...core.utils import log_exceptions

torch.multiprocessing.set_sharing_strategy('file_system')
ImageFile.LOAD_TRUNCATED_IMAGES = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FSLModel(ModelBase):

    def __init__(self,
                 args: Dict,
                 model_path: Path,
                 version: int,
                 mode: str,
                 context: ModelContext,
                 support_path: str):

        logger.info(f"Loading FSL Model from {model_path}")
        assert model_path.is_file()
        args['input_size'] = args.get('input_size', 256)

        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        args['test_batch_size'] = args.get('test_batch_size', 64)
        args['version'] = version
        args['arch'] = args.get('arch', 'siamese')
        args['train_examples'] = args.get('train_examples', {'1':0, '0':0})
        args['mode'] = mode

        self.args = args

        super().__init__(self.args, model_path, context)

        self._train_examples = args['train_examples']
        self._test_transforms = test_transforms
        self._batch_size = args['test_batch_size']

        self._model = self.load_model(model_path)
        self._device = device
        self._model.to(self._device)
        self._model.eval()
        self._running = True

        support = Image.open(support_path).convert('RGB')
        self.support = self.get_embed(support)


    def get_embed(self, im):
        im = im.resize((224,224))
        im = torch.unsqueeze(self._test_transforms(im), dim=0)
        with torch.no_grad():
            preds = self._model(im.to(device))
            preds = np.array([preds[0].cpu().numpy()])
            return preds

    @property
    def version(self) -> int:
        return self._version

    def preprocess(self, request: ObjectProvider) -> Tuple[ObjectProvider, torch.Tensor]:
        embed = []
        try:
            image = Image.open(io.BytesIO(request.content))

            if image.mode != 'RGB':
                image = image.convert('RGB')
            embed = self.get_embed(image)
        except Exception as e:
            raise(e)

        return request, embed

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
        logger.info("Starting model load")
        model = models.resnet18().cuda()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        logger.info("Starting model complete")
        return model

    def get_predictions(self, inputs: torch.Tensor) -> List[float]:
        probability = []
        with torch.no_grad():
            similarity = cosine_similarity(self.support, inputs)
            if similarity.shape[-1] == 1:
                similarity = [float(similarity)]
            else:
                similarity = np.squeeze(similarity)
            return similarity


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


    def _process_batch(self, batch: List[Tuple[ObjectProvider, torch.Tensor]]) -> Iterable[ResultProvider]:
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
                if self._mode == "oracle":
                    if '/0/' in batch[i][0].id:
                        score = 0
                    else:
                        score = 1
                batch[i][0].attributes.add({'score': str.encode(str(score))})
                yield ResultProvider(batch[i][0], score, self.version)

    def stop(self):
        logger.info(f"Stopping model of version {self.version}")
        with self._model_lock:
            self._running = False
            self._model = None
