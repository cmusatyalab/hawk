# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only
from __future__ import annotations

import itertools
import time
from typing import TYPE_CHECKING

import numpy as np
from logzero import logger
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..core.utils import log_exceptions
from .topk_selector import TopKSelector

if TYPE_CHECKING:
    from ..core.result_provider import ResultProvider
    from ..reexamination.reexamination_strategy import ReexaminationStrategy


class DiversitySelector(TopKSelector):
    def __init__(
        self,
        mission_id: str,
        k: int,
        batch_size: int,
        countermeasure_threshold: float,
        total_countermeasures: int,
        reexamination_strategy: ReexaminationStrategy,
        add_negatives: bool = False,
    ) -> None:
        assert k < batch_size
        super().__init__(
            mission_id,
            k,
            batch_size,
            countermeasure_threshold,
            total_countermeasures,
            reexamination_strategy,
            add_negatives,
        )

        self._div_k = int(k / 3)
        self._k = k - self._div_k
        self._result_list: list[ResultProvider] = []

        self._model = None
        self.n_pca = 5
        self.min_sample = 3

    def diversity_sample(self) -> list[ResultProvider]:
        logger.info("Diversity start")
        objects = []
        original = np.array(list(self._result_list))
        results: list[ResultProvider] = []

        for result in self._result_list:
            objects.append(result)

        logger.info("Starting embeddings")
        if self._model is None:
            return results

        embeddings = self._model.get_embeddings(objects)
        logger.info("Found embeddings")

        # Dimensionality reduction
        embeddings = StandardScaler().fit_transform(embeddings)
        pca = PCA(self.n_pca)
        embeddings = pca.fit_transform(embeddings)

        # Density clustering
        cluster_learner = DBSCAN(eps=2, min_samples=self.min_sample)

        # cluster_idxs = cluster_learner.fit_predict(embeddings)
        cluster_learner.fit_predict(embeddings)

        logger.info("Found clusters")
        data_labels = cluster_learner.labels_
        unique_labels = set(data_labels)

        # get the number of clusters
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        cluster_labels = np.arange(num_clusters)
        np.random.shuffle(cluster_labels)
        logger.info(f"Num clusters {num_clusters}")
        n = self._div_k
        if num_clusters <= self._div_k:
            num_samples = [
                (n // num_clusters) + (1 if i < (n % num_clusters) else 0)
                for i in range(num_clusters)
            ]
        else:
            num_samples = [1] * n

        q_idxs = set()
        # sampled = 0
        cluster_label_gen = itertools.cycle(cluster_labels)

        for num_sample in num_samples:
            label = next(cluster_label_gen)
            n_c = np.where(data_labels == label)[0]
            n_ = len(n_c)
            logger.info(f"{label} {n_} {num_sample}")
            n_sample = min(n_, num_sample)
            sample_idxs = list(np.random.choice(n_c, n_sample, replace=False))
            q_idxs = q_idxs | set(sample_idxs)

        q_idxs = np.array(list(q_idxs))
        q_idxs = q_idxs.astype(int)

        # len_array = len(original)
        return original[q_idxs]

    @log_exceptions
    def select_tiles(self, _num_tiles: int) -> None:
        assert self._mission is not None

        # TopK sampling
        results = []
        logger.info("TopK call")
        for i in range(self._k):
            result = self._priority_queues.get()[-1]
            self.priority_queue_length.dec()
            self._mission.log(
                f"{self.version} {i}_{self._k} SEL: FILE SELECTED {result.id}",
            )
            if not self._is_oracle:
                # self.result_queue_length.inc()
                # self.result_queue.put(result)
                results.append(result)

        self._result_list = list(set(self._result_list) - set(results))

        # diversity sampling
        time_start = time.time()
        div_sample = self.diversity_sample()
        logger.info(f"Time taken {time.time() - time_start}")
        results += list(div_sample)
        for result in results:
            self.result_queue_length.inc()
            self.result_queue.put(result)
            logger.info(f"[Result] Id {result.id} Score {result.score}")

        self._batch_added -= self._batch_size
