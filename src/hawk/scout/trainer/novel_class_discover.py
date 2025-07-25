# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import io
import operator
import os
import random
import threading
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import torch
from logzero import logger
from sklearn.cluster import KMeans, kmeans_plusplus

if TYPE_CHECKING:
    import multiprocessing as mp
    from multiprocessing.connection import _ConnectionBase

    from ..core.result_provider import ResultProvider
    from ..retrieval.retriever import Retriever


class Novel_Class_Clustering:
    def __init__(
        self,
        retriever: Retriever,
        inbound_result_queue: mp.Queue[ResultProvider],
        labels_queue: mp.Queue[tuple[str, Path]],
        outbound_pipe: _ConnectionBase,
        fv_dir: Path,
        scout_index: int,
        semi_supervised: bool = False,
    ) -> None:
        self.retriever = retriever
        self.fv_dir = fv_dir
        self.inbound_result_queue = inbound_result_queue
        self.labels_queue = labels_queue
        self.feature_vector = None
        self.scout_index = scout_index
        self.cluster_iteration = threading.Event()
        self.outbound_pipe = outbound_pipe
        self.batch_sample_count = 0
        self.labeled_samples_dict: dict[str, list[Path]] = defaultdict(list)
        self.unlabeled_result_list: list[tuple[float, ResultProvider]] = []
        self.threshold_samples_for_clustering: int = (
            500  # cluster every N samples received from model
        )
        self.max_unlabeled_for_clustering = (
            100  # max number of unlabeled used for clustering every cluster iteration
        )
        self.num_selected_samples_per_iter: int = (
            5  ## number of samples to send to home after each clustering
        )
        self.unlabeled_selection_mode: str = "top"  ## or random
        self.n_clusters = 5
        self.random_state = 42
        ## True if wanting to leverage existing labels as separate cluster
        ## groups, False for normal KMeans for only unlabeled samples
        self.semi_supervised = semi_supervised
        self.unlabeled_list_lock = threading.Lock()
        self.start_end_index_labeled: dict[str, tuple[int, int]] = {}
        self.sample_selection = "random"  ## or top from closest samples to class 1

        threading.Thread(target=self.clustering_thread).start()
        threading.Thread(target=self.receive_labels_thread).start()

    def euclidean_distance(
        self,
        point1: npt.NDArray[Any],
        point2: npt.NDArray[Any],
    ) -> float:
        return cast("float", np.linalg.norm(point1 - point2))

    def novel_clustering_process(self) -> None:
        logger.info("Running novel class discovery...")
        self.unlabeled_sample_count = 0

        while True:
            ## this loop manages the incoming unlabeled samples from the queue
            ## and releases the clustering thread when n new samples have been
            ## inferenced.
            with self.unlabeled_list_lock:
                ## ensure no new unlabeled samples can be retrieved while the
                ## current list of unlabeled samples is being processed in the
                ## clustering thread.  As soon as the list of samples for
                ## clustering is set, release other lock.
                result: ResultProvider = (
                    self.inbound_result_queue.get()
                )  ## grab each result provider object
                self.unlabeled_result_list.append((-result.score, result))
                self.batch_sample_count += 1

                ## if condition is met, set the clustering event.
                if self.batch_sample_count == self.threshold_samples_for_clustering:
                    self.cluster_iteration.set()  ## trigger clustering
                    self.batch_sample_count = 0  ## reset batch counter

    def clustering_thread(self) -> None:
        while True:
            self.cluster_iteration.wait()
            logger.info("Running periodic clustering thread...\n")
            ## select unlabeled samples
            with self.unlabeled_list_lock:
                ## execute this block before any new unlabeled samples can be
                ## added to the current list, then reset the list
                ## Before sorting and slicing or randomly selecting, make sure
                ## to check if any of these unlabeled sample have been converted
                ## to labeled samples (i.e. they have been labeled in the time
                ## since they were received by this queue)
                self.unlabeled_result_list.sort(
                    key=lambda x: x[0],
                )  ## sort the current state of the unlabeled results list
                if self.unlabeled_selection_mode == "top":
                    ## get the top N highest-scoring samples
                    unlabeled_set_for_clustering = self.unlabeled_result_list[
                        : self.max_unlabeled_for_clustering
                    ]
                else:
                    ## pick a random set of n samples from all of the unlabeled
                    ## samples (more likely to get Far OOD samples)
                    unlabeled_set_for_clustering = random.sample(
                        self.unlabeled_result_list,
                        min(
                            len(self.unlabeled_result_list),
                            self.max_unlabeled_for_clustering,
                        ),
                    )
                self.unlabeled_result_list = []

            ## dict with id as key and feature vector as value
            self.unlabeled_list_by_id = {
                unlabeled_sample[1].id: unlabeled_sample[1]
                for unlabeled_sample in unlabeled_set_for_clustering
            }
            unlabeled_result_obj = list(self.unlabeled_list_by_id.values())

            unlabeled_feature_vectors = torch.stack(
                [
                    torch.load(
                        # When novel class discovery is enabled we should have fv
                        io.BytesIO(result.feature_vector),  # type: ignore[arg-type]
                    )
                    for result in unlabeled_result_obj
                ],
            ).numpy()  ## numpy stack of vectors for clustering

            ## choose whether to perform default kmeans or semi-supervised kmeans
            if not self.semi_supervised:
                sklearn_kmeans = KMeans(
                    n_clusters=self.n_clusters,
                    init="k-means++",
                    random_state=self.random_state,
                    n_init=1,
                )
                sklearn_kmeans.fit(
                    unlabeled_feature_vectors,
                )  ## All default KMeans calls.
                cluster_labels = sklearn_kmeans.labels_
                cluster_centroids = sklearn_kmeans.cluster_centers_
            else:
                ## semi-supervised approach...
                ## need to track labeled samples for clustering
                label_dirs = sorted(self.labeled_samples_dict.keys())
                logger.info(f"Label dir keys: {label_dirs}")
                for label_dir in label_dirs:
                    if label_dir == "0":
                        start_index = len(unlabeled_result_obj)
                        end_index = len(unlabeled_result_obj) + len(
                            self.labeled_samples_dict[label_dir],
                        )
                    else:
                        start_index = end_index
                        end_index = start_index + len(
                            self.labeled_samples_dict[label_dir],
                        )
                    self.start_end_index_labeled[label_dir] = (start_index, end_index)
                num_labels_by_dir = [
                    len(os.listdir(Path(self.fv_dir) / label_dir))
                    for label_dir in label_dirs
                ]
                must_link, cannot_link = self.create_constraints_multiple_groups(
                    num_labels_by_dir,
                )  ## generate constraint tuples in and across labeled samples

                ## create the labeled feature vectors from paths in label dirs
                self.all_labeled_fvs = torch.stack(
                    [
                        torch.load(base)
                        for label_dir in label_dirs
                        for base in self.labeled_samples_dict[label_dir]
                    ],
                ).numpy()

                ## concatenate unlabeled and labeled feature vectors for clustering
                self.semi_super_all_vectors = np.concatenate(
                    (unlabeled_feature_vectors, self.all_labeled_fvs),
                    axis=0,
                )

                cluster_labels, cluster_centroids = self.semi_kmeans(
                    unlabeled_feature_vectors,
                    n_clusters=self.n_clusters,
                    random_state=self.random_state,
                    must_link=must_link,
                    cannot_link=cannot_link,
                )
                ## only remaining function to add for semi-supervised: instead
                ## of selecting n samples from each cluster, select all samples
                ## from the cluster centroid closest to one of the positive
                ## classes.  But, the selected samples must be from unlabeled
                ## samples.  So, find the clusters with centroids closest to
                ## positive classes, and find samples from that cluster in the
                ## unlabeled samples.

            ## After clustering of either type, choose which samples should be
            ## tiled and sent to home, e.g. 2 samples from each cluster or n
            ## samples from closest cluster to positive class, e.g. 1, need add
            ## this functionality to select_sample_labels.
            selected_sample_indices = self.select_sample_labels(cluster_labels)

            ## Final step: send selected tiles to home.  Future: write
            ## clustering results to scout for each iter.
            self.send_samples(
                list(
                    operator.itemgetter(*selected_sample_indices)(unlabeled_result_obj),
                ),
            )

            self.cluster_iteration.clear()

    def send_samples(self, selected_samples: list[ResultProvider]) -> None:
        ## For each selected sample, create tile and send to home
        for i, selected_sample in enumerate(selected_samples):
            tile = selected_sample.to_protobuf(
                self.retriever,
                self.scout_index,
                novel_sample=True,
            )
            ## the cluster label integer: if 1 is first cluster, 2 is second cluster
            ## cluster assignment
            ## cluster iteration source

            ## send each sample to home
            self.outbound_pipe.send(tile.SerializeToString())
            logger.info(f" Sent item {i} to home...\n")

    def receive_labels_thread(self) -> None:
        while True:
            logger.info("Waiting in label thread...")
            label, labeled_fv_path = self.labels_queue.get()
            ## append the paths of each feature vector according to the dict
            ## according to their label.  Don't actually load until ready to
            ## cluster.
            self.labeled_samples_dict[label].append(labeled_fv_path)
            # logger.info(f"Got a label: {label_path}")

    def select_sample_labels(
        self,
        labels: list[str],
        num_samples_per_label: int = 2,
    ) -> list[bool]:
        """Randomly samples a indices for each unique label in a list.

        Args:
            labels: A list of labels.
            num_samples_per_label: The number of samples to randomly select for
              each label.

        Returns:
            A list of indices of the sampled elements. Returns an empty list if
            there are fewer samples than requested for a label.

        """
        unique_labels = np.unique(labels)
        sampled_indices: list[bool] = []

        for label in unique_labels:
            label_indices = np.where(labels == label)[
                0
            ]  # Get indices of elements with the current label
            num_available = len(label_indices)

            if num_available == 0:
                print(f"Warning: Label {label} has no samples.")
                continue  # Skip if no samples for this label

            # Sample all available if fewer than requested
            num_to_sample = min(num_samples_per_label, num_available)

            # Create a local random number generator to avoid side effects
            rng = np.random.RandomState()
            rng.shuffle(label_indices)  # Shuffle the indices randomly

            # Select the appropriate number of indices
            sampled_indices.extend(label_indices[:num_to_sample])

        return sampled_indices

    def create_constraints_multiple_groups(
        self,
        group_sizes: list[int],
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Creates must-link and cannot-link constraints for multiple groups.

        Args:
            group_sizes: A list of integers representing the number of samples
              in each group.

        Returns:
            A tuple containing two lists: (must_link, cannot_link).

        """
        must_link = []
        cannot_link = []
        total_samples = 0

        for group_index, group_size in enumerate(group_sizes):
            # Must-link constraints within the current group
            if group_index != 0:
                for i in range(group_size):
                    for j in range(i + 1, group_size):
                        must_link.append((total_samples + i, total_samples + j))

            # Cannot-link constraints between the current group and all previous groups
            for prev_group_index in range(group_index):
                prev_group_size = group_sizes[prev_group_index]
                for i in range(group_size):
                    for j in range(prev_group_size):
                        cannot_link.append(
                            (
                                total_samples + i,
                                sum(group_sizes[:prev_group_index]) + j,
                            ),
                        )

            total_samples += group_size

        return must_link, cannot_link

    def semi_kmeans(
        self,
        vectors: npt.NDArray[Any],
        n_clusters: int,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
        must_link: list[tuple[int, int]] | None = None,
        cannot_link: list[tuple[int, int]] | None = None,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """K-Means from scratch using sklearn's k-means++ initialization."""
        centroids, _ = kmeans_plusplus(vectors, n_clusters, random_state=random_state)

        n_samples = vectors.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        if must_link is None:
            must_link = []
        if cannot_link is None:
            cannot_link = []

        for _ in range(max_iter):
            new_labels = labels.copy()
            for i in range(n_samples):
                possible_labels = list(range(n_clusters))
                for j in range(n_samples):
                    if labels[j] in possible_labels and (
                        (i, j) in cannot_link or (j, i) in cannot_link
                    ):
                        possible_labels.remove(labels[j])
                if len(possible_labels) == 0:
                    continue

                ## compute distances from point to all possible centroids that
                ## are not prevented by constraint (sample with label 0 cannot
                ## be assigned to the same cluster as a sample with label 1)
                distances = np.array(
                    [
                        self.euclidean_distance(vectors[i], centroids[k])
                        for k in possible_labels
                    ],
                )
                best_label_index = np.argmin(distances)
                new_labels[i] = possible_labels[best_label_index]

            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(n_clusters)
            for k in range(n_clusters):
                cluster_points = vectors[new_labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = cluster_points.mean(axis=0)
                    counts[k] += len(cluster_points)
            if np.allclose(centroids, new_centroids, atol=tol):
                break
            centroids = new_centroids
            labels = new_labels

        return labels, centroids


def main(
    retriever: Retriever,
    result_queue: mp.Queue[ResultProvider],
    labels_queue: mp.Queue[tuple[str, Path]],
    home_pipe: _ConnectionBase,
    clustering_dir: Path,
    scout_index: int,
) -> None:
    ### create clustering object
    novel_clustering = Novel_Class_Clustering(
        retriever,
        result_queue,
        labels_queue,
        home_pipe,
        clustering_dir,
        scout_index,
        # semi_supervised=True,
    )
    novel_clustering.novel_clustering_process()
