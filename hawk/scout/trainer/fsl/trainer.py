# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import json
import os
import glob
import time
import random
import numpy as np
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from logzero import logger
from torch_snippets import *


import glob
import subprocess

from ...context.model_trainer_context import ModelContext
from ...core.model import Model
from ...core.model_trainer import ModelTrainerBase
from . import PYTHON_EXEC
from .model import FSLModel
from .utils import TripletData, TripletLoss

torch.multiprocessing.set_sharing_strategy('file_system')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FSLTrainer(ModelTrainerBase):

    def __init__(self, context: ModelContext, args: Dict[str, str]):
        super().__init__(args)
        
        assert 'support_path' in self.args
        assert 'fsl_traindir' in self.args

        self.args['test_dir'] = self.args.get('test_dir', '')
        self.args['arch'] = self.args.get('arch', 'siamese')
        self.args['batch-size'] = int(self.args.get('batch-size', 64))
        self.args['initial_model_epochs'] = int(self.args.get('initial_model_epochs', 15))
        self.train_initial_model = False 
        self.testpath = self.args['test_dir']

        self.context = context

        logger.info(f"Model_dir {self.context.model_dir}")

        logger.info("FSL TRAINER CALLED")
        
    def load_model(self, path:Path = "", content:bytes = b'', version: int = -1):
        if isinstance(path, str):
            path = Path(path)

        new_version = self.get_new_version()

        assert path.is_file() or len(content)
        if not path.is_file():
            path = self.context.model_path(new_version)
            path.write_bytes(content)
            
        version = self.get_version()
        logger.info("Loading from path {}".format(path))
        self.prev_path = path
        return FSLModel(self.args, path, version,  
                                  mode=self.args['mode'],
                                  context=self.context,
                                  support_path= self.args['support_path']) 
    
    def train_batch(self, model, data, optimizer, criterion):
        imgsA, imgsB, labels = [t.to(device) for t in data]
        optimizer.zero_grad()
        codesA, codesB = model(imgsA, imgsB)
        loss, acc = criterion(codesA, codesB, labels)
        loss.backward()
        optimizer.step()
        return loss.item(), acc.item() 
    
    
    def train_model(self, train_dir='') -> Model:
        
        new_version = self.get_new_version()

        model_savepath = self.context.model_path(new_version)
       
        file_path = os.path.dirname(os.path.abspath(__file__))
        train_dataset = self.args['fsl_traindir']
        support_path = self.args['support_path']
        
        cmd = [
            PYTHON_EXEC,
            f"{file_path}/augment.py",
            train_dataset,
            support_path,
        ]
        proc = subprocess.Popen(cmd)
        proc.communicate()

        train_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        
        train_data = TripletData(train_dataset, train_transforms)
        train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=32, shuffle=True, num_workers=4)

        device = 'cuda'

        # Our base model
        model = models.resnet18().cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        triplet_loss = TripletLoss()

        # Training
        time_start = time.time()
        epochs = 2
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for data in train_loader:
                optimizer.zero_grad()
                x1,x2,x3 = data
                e1 = model(x1.to(device))
                e2 = model(x2.to(device))
                e3 = model(x3.to(device)) 
                
                loss = triplet_loss(e1,e2,e3)
                epoch_loss += loss
                loss.backward()
                optimizer.step()
            logger.info("Train Loss: {} {}".format(epoch, epoch_loss.item()))

        time_end = time.time()
        logger.info(f"Training time = {time_end - time_start}")

        torch.save(model.state_dict(), model_savepath)
        # train completed time 
        train_time = time_end - time_start
        
        self.prev_path = model_savepath

        model_args = self.args.copy()
        model_args['train_examples'] = {'1':1, '0':0}
        model_args['train_time'] = train_time
        
        return FSLModel(model_args, 
                        model_savepath,
                        new_version,
                        self.args['mode'], 
                        context=self.context, 
                        support_path=self.args['support_path'])
