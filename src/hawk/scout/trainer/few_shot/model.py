# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import io
import os
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from logzero import logger
from PIL import Image, ImageFile

from ...context.model_trainer_context import ModelContext
from ...core.model import ModelBase
from ...core.object_provider import ObjectProvider
from ...core.result_provider import ResultProvider
from ...core.utils import log_exceptions
from .models.SnaTCHerF import SnaTCHerF

torch.multiprocessing.set_sharing_strategy("file_system")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Source: https://github.com/MinkiJ/SnaTCHer


class FewShotModel(ModelBase):
    def __init__(
        self,
        args: Dict,
        model_path: Path,
        version: int,
        mode: str,
        context: ModelContext,
    ):
        # model_path is None for few-shot
        # Hardcoding model architecture to resnet-50
        logger.info(f"Loading FSL Model from {model_path}")
        test_transforms = transforms.Compose(
            [
                transforms.Resize(92),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                transforms.Normalize(
                    np.array(
                        [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
                    ),
                    np.array(
                        [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
                    ),
                ),
            ]
        )

        args["version"] = version
        args["train_examples"] = args.get("train_examples", {"1": 0, "0": 0})
        args["mode"] = mode
        self.args = args

        super().__init__(self.args, model_path, context)

        self._train_examples = args["train_examples"]
        self._test_transforms = test_transforms
        self._model = self.load_model(model_path)
        self._device = torch.device("cpu")
        self._model.to(self._device)
        self._model.eval()
        self.train_data = self.args["support_data"]
        self.load_supports()
        self._running = True

    def unzip_support(self):
        self.train_dir = "/tmp/supports"
        labels = set()
        with zipfile.ZipFile(io.BytesIO(self.args["support_data"]), "r") as zf:
            example_files = zf.namelist()
            for filename in example_files:
                basename = Path(filename).name
                parent_name = Path(filename).parent.name
                label = parent_name
                self.labels.add(label)
                content = zf.read(filename)
                path = os.path.join(self.train_dir, label, basename)
                if not os.path.exists(Path(path).parent.name):
                    os.makedirs(Path(path).parent.name, exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(content)
        assert len(labels) == 5, "Incompatible n_shot {len(labels)}"

    def load_supports(self):
        self.unzip_support()
        trainset = torchvision.datasets.ImageFolder(
            self.train_dir, transform=self._test_transforms
        )
        train_loader = torch.utils.data.DataLoader(dataset=trainset, pin_memory=True)
        batch = next(iter(train_loader))
        data, _ = (_ for _ in batch)
        _ = self._model(data)
        instance_embs = self._model.probe_instance_embs
        support_shape = (1, 5, 5)
        support = instance_embs.view(*(support_shape + (-1,)))
        logger.info(f"Support emb shape {support.shape}")
        self.emb_dim = support.shape[-1]
        support = support.contiguous()
        self.proto = support.mean(dim=1)  # Ntask x NK x d
        s_proto = proto
        s_proto = self._model.slf_attn(s_proto, s_proto, s_proto)
        s_proto = s_proto[0]
        self.support_proto = s_proto

    @property
    def version(self) -> int:
        return self._version

    def preprocess(
        self, request: ObjectProvider
    ) -> Tuple[ObjectProvider, torch.Tensor]:
        try:
            image = Image.open(io.BytesIO(request.content))

            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            raise (e)

        return request, self._test_transforms(image)

    def serialize(self) -> bytes:
        if self._model is None:
            return None

        content = io.BytesIO()
        torch.save(
            {
                "state_dict": self._model.state_dict(),
            },
            content,
        )
        content.seek(0)

        return content.getvalue()

    def load_model(self, model_path: Path):
        from argparse import Namespace

        args = Namespace(
            backbone_class="Res12", closed_way=5, gpu="0", multi_gpu=False, shot=5
        )
        model = SnaTCHerF(args)
        weights = torch.load(model_path)
        model_weights = weights["params"]
        model.load_state_dict(model_weights, strict=False)
        return model

    def get_predictions(self, inputs: torch.Tensor) -> List[float]:
        if len(data.shape) == 3:
            data = torch.unsqueeze(data, dim=0)
        _ = self._model(data)
        query = self._model.probe_instance_embs.contiguous()
        qlogits = (
            -(query.reshape(-1, 1, self.emb_dim) - self.support_proto).pow(2).sum(2)
            / 64.0
        )
        batch_size = data.shape[0]
        predictions = []
        with torch.no_grad():
            for j in range(batch_size):
                pproto = self.proto.clone().detach()
                # c = qlogits.argmax(1)[j]
                # Assuming class 0 is the needed class ; c = 0
                c = 0
                pproto[0][c] = query.reshape(-1, self.emb_dim)[j]

                pproto = self._model.slf_attn(pproto, pproto, pproto)[0]
                pdiff = (pproto - self.support_proto).pow(2).sum(-1).sum() / 64.0

                predictions.append(pdiff)

        pkdiff = torch.stack(predictions)
        pkdiff = pkdiff.cpu().detach().numpy()
        return pkdiff

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

            if (
                len(requests) >= self._batch_size
                or (time.time() - prev_infer) > timeout
            ):
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
            for request in requests[i : i + self._batch_size]:
                batch.append(self.preprocess(request))
            results = self._process_batch(batch)
            for result in results:
                output.append(result)

        return output

    def _process_batch(
        self, batch: List[Tuple[ObjectProvider, torch.Tensor]]
    ) -> Iterable[ResultProvider]:
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
                batch[i][0].attributes.add({"score": str.encode(str(score))})
                yield ResultProvider(batch[i][0], score, self.version)

    def stop(self):
        logger.info(f"Stopping model of version {self.version}")
        with self._model_lock:
            self._running = False
            self._model = None
