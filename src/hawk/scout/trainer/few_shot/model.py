# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only
from __future__ import annotations

import io
import queue
import time
import zipfile
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
import torch
import torchvision
from logzero import logger
from PIL import Image, ImageFile
from torchvision import transforms

from ....classes import POSITIVE_CLASS
from ....detection import Detection
from ...core.model import ModelBase
from ...core.result_provider import ResultProvider
from ...core.utils import log_exceptions
from .models.SnaTCHerF import SnaTCHerF

if TYPE_CHECKING:
    from ....hawkobject import HawkObject
    from ....objectid import ObjectId
    from ...context.model_trainer_context import ModelContext
    from .config import FewShotModelConfig

torch.multiprocessing.set_sharing_strategy("file_system")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Source: https://github.com/MinkiJ/SnaTCHer


class FewShotModel(ModelBase):
    config: FewShotModelConfig

    def __init__(
        self,
        config: FewShotModelConfig,
        context: ModelContext,
        model_path: Path,
        version: int,
        *,
        train_examples: dict[str, int] | None = None,
        train_time: float = 0.0,
    ) -> None:
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
                        [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                    ),
                    np.array(
                        [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]],
                    ),
                ),
            ],
        )

        super().__init__(
            context,
            config,
            model_path,
            version,
            train_examples,
            train_time,
        )

        self._test_transforms = test_transforms
        self._model = self.load_model(model_path)
        self._device = torch.device("cpu")
        self._model.to(self._device)
        self._model.eval()

        self.load_supports()
        self._running = True

    def unzip_support(self) -> None:
        self.train_dir = Path("/tmp/supports")
        labels: set[str] = set()
        with io.BytesIO(self.config.support_data) as f, zipfile.ZipFile(f, "r") as zf:
            example_files = zf.namelist()
            for filename in example_files:
                basename = Path(filename).name
                parent_name = Path(filename).parent.name
                label = parent_name
                labels.add(label)
                content = zf.read(filename)

                path = self.train_dir / label / basename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        assert len(labels) == 5, "Incompatible n_shot {len(labels)}"

    def load_supports(self) -> None:
        self.unzip_support()
        trainset = torchvision.datasets.ImageFolder(
            self.train_dir,
            transform=self._test_transforms,
        )
        train_loader = torch.utils.data.DataLoader(dataset=trainset, pin_memory=True)
        batch = next(iter(train_loader))
        data, _ = (_ for _ in batch)
        _ = self._model(data)
        instance_embs = self._model.probe_instance_embs
        support_shape = (1, 5, 5)
        support = instance_embs.view(*((*support_shape, -1)))
        logger.info(f"Support emb shape {support.shape}")
        self.emb_dim = support.shape[-1]
        support = support.contiguous()
        self.proto = support.mean(dim=1)  # Ntask x NK x d
        s_proto = self.proto
        s_proto = self._model.slf_attn(s_proto, s_proto, s_proto)
        s_proto = s_proto[0]
        self.support_proto = s_proto

    def preprocess(self, obj: HawkObject) -> torch.Tensor:
        assert obj.media_type.startswith("image/")

        image = Image.open(io.BytesIO(obj.content))
        if image.mode != "RGB":
            image = image.convert("RGB")

        return self._test_transforms(image)

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

    def load_model(self, model_path: Path) -> SnaTCHerF:
        args = Namespace(
            backbone_class="Res12",
            closed_way=5,
            gpu="0",
            multi_gpu=False,
            shot=5,
        )
        model = SnaTCHerF(args)
        weights = torch.load(model_path)
        model_weights = weights["params"]
        model.load_state_dict(model_weights, strict=False)
        return model

    def get_predictions(self, inputs: torch.Tensor) -> list[float]:
        if len(inputs.shape) == 3:
            inputs = torch.unsqueeze(inputs, dim=0)
        _ = self._model(inputs)
        query = self._model.probe_instance_embs.contiguous()
        # qlogits = (
        #    -(query.reshape(-1, 1, self.emb_dim) - self.support_proto).pow(2).sum(2)
        #    / 64.0
        # )
        batch_size = inputs.shape[0]
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
        return pkdiff.cpu().detach().numpy()

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
        self,
        batch: list[tuple[ObjectId, torch.Tensor]],
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
                bboxes = [Detection(class_name=POSITIVE_CLASS, confidence=score)]
                yield ResultProvider(batch[i][0], score, bboxes, self.version)

    def stop(self) -> None:
        logger.info(f"Stopping model of version {self.version}")
        with self._model_lock:
            self._running = False
            self._model = None
