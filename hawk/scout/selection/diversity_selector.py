# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import copy
import itertools
import os
import queue
import random
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from logzero import logger
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..core.model import Model
from ..core.result_provider import ResultProvider
from ..core.utils import get_example_key, log_exceptions
from ..reexamination.reexamination_strategy import ReexaminationStrategy
from .selector_base import SelectorBase, SelectorStats
from .topk_selector import TopKSelector


class DiversitySelector(TopKSelector):

    def __init__(self, k: int, batch_size: int,
                 reexamination_strategy: ReexaminationStrategy,
                 add_negatives: bool = False):
        assert k < batch_size
        super().__init__()

        self.version = 0
        self.model_train_time = 0
        self._div_k = int(k/3)
        self._k = k - self._div_k
        self._result_list = []

        self._batch_size = batch_size
        self._reexamination_strategy = reexamination_strategy

        self._priority_queues = [queue.PriorityQueue()]
        self._batch_added = 0
        self._insert_lock = threading.Lock()
        self._mode = "hawk"
        self._model = None
        self.n_pca = 5
        self.min_sample = 3
        self.log_counter = [int(i/3.*self._batch_size) for i in range(1, 4)]

    def diversity_sample(self):
        logger.info("Diversity start")
        objects = []
        original = np.array(list(self._result_list))
        results = []
        
        for result in self._result_list:
            objects.append(result)

        logger.info("Starting embeddings")
        if self._model is not None:
            embeddings = self._model.get_embeddings(objects)
        else:
            return results
        logger.info("Found embeddings")

        #Dimensionality reduction
        embeddings = StandardScaler().fit_transform(embeddings)
        pca = PCA(self.n_pca)
        embeddings = pca.fit_transform(embeddings)

        #Desnity clustering
        cluster_learner = DBSCAN(eps=2, min_samples=self.min_sample)
        cluster_idxs = cluster_learner.fit_predict(embeddings)
        logger.info("Found clusters")
        data_labels = cluster_learner.labels_
        unique_labels = set(data_labels)

        # get the number of clusters
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        cluster_labels = np.arange(num_clusters)
        np.random.shuffle(cluster_labels)
        logger.info("Num clusters {}".format(num_clusters))
        n = self._div_k
        if num_clusters <= self._div_k:
            num_samples = [(n // num_clusters) + (1 if i < (n % num_clusters) else 0) for i in range(num_clusters)]
        else:
            num_samples = [1]*n

        q_idxs = set()
        sampled = 0
        cluster_label_gen = itertools.cycle(cluster_labels)

        for num_sample in num_samples:
            label = next(cluster_label_gen)
            n_c = np.where(data_labels==label)[0]
            n_ = len(n_c)
            logger.info("{} {} {}".format(label, n_, num_sample))
            n_sample = min(n_, num_sample)
            sample_idxs = list(np.random.choice(n_c, n_sample, replace=False))
            q_idxs = q_idxs | set(sample_idxs)

        q_idxs = np.array(list(q_idxs))
        q_idxs = q_idxs.astype(int)
        
        len_array = len(original) 
        results =  original[q_idxs]
        
        return results


    @log_exceptions
    def select_tiles(self):
        #TopK sampling
        results = []
        logger.info("TopK call")
        for i in range(self._k):
            result = self._priority_queues[-1].get()[-1]
            if self._mission.enable_logfile:
                self._mission.log_file.write("{:.3f} {}_{} {}_{} SEL: FILE SELECTED {}\n".format(
                    time.time() - self._mission.start_time, self._mission.host_name,
                    self.version, i, self._k, result.id))
                if self._mode != "oracle":
                    # self.result_queue.put(result)
                    results.append(result)
                
        self._result_list = list(set(self._result_list) - set(results))

        # diversity sampling 
        time_start = time.time()
        div_sample = self.diversity_sample()
        logger.info("Time taken {}".format(time.time() - time_start))
        results += list(div_sample)
        for result in results:
            self.result_queue.put(result)
            logger.info("[Result] Id {} Score {}".format(result.id, result.score))

        self._batch_added -= self._batch_size
         
