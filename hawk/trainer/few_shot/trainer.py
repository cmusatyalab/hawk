# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import json
import os
import glob
import time
from pathlib import Path
from typing import Dict

import torch
import torchvisio
from logzero import logger

import shlex
import subprocess

from hawk import M_ZFILL
from hawk.context.model_trainer_context import ModelTrainerContext
from hawk.core.model_trainer import ModelTrainerBase
from hawk.core.model import Model
from hawk.trainer.dnn_classifier import PYTHON_EXEC
from hawk.trainer.few_shot.model import FewShotModel

torch.multiprocessing.set_sharing_strategy('file_system')


class FewShotTrainer(ModelTrainerBase):

    def __init__(self, context: ModelTrainerContext, args: Dict[str, str]):
        assert 'support_data' in args
        super().__init__(args)
        
        self.context = context
        self._model_dir = self.context.model_dir
        self._model_path = None
        self.version = 0
        
        logger.info("FSL TRAINER CALLED")
            

    def load_model(self, path:Path = "", content:bytes = b'', version: int = -1):
        if isinstance(path, str):
            path = Path(path)
        assert path.is_file() or len(content)
        if not path.is_file():
            path = self._model_dir / "model-{}.pth".format(
                str(new_version).zfill(M_ZFILL))  
            with open(path, "wb") as f:
                f.write(content)     
            
        logger.info("Loading from path {}".format(path))
        self._model_path = path
        return FewShotModel(self.args, path, version,  
                                  context=self.context) 

    def train_model(self, train_dir) -> Model:
        return self.load_model(self._model_path, version=self.version) 
        
