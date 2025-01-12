# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import time
import queue
import random
import os
import operator
from pathlib import Path
from collections import defaultdict
import torch, io
import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus
from scipy.optimize import linear_sum_assignment
from logzero import logger
import multiprocessing as mp
from ..core.result_provider import ResultProvider
from multiprocessing.connection import _ConnectionBase
import threading
from typing import Tuple
from ...classes import ClassCounter, ClassName, class_name_to_str
from ...proto.messages_pb2 import (
    BoundingBox,
    SendTile
)


class Novel_Class_Clustering:
    def __init__(self, inbound_result_queue: mp.Queue, labels_queue: mp.Queue, outbound_pipe: _ConnectionBase, fv_dir: Path, scout_index: int, semi_supervised=False):
        self.fv_dir = fv_dir
        self.inbound_result_queue = inbound_result_queue
        self.labels_queue = labels_queue
        self.feature_vector = None
        self.scout_index = scout_index
        self.cluster_iteration = threading.Event()
        self.outbound_pipe = outbound_pipe
        self.batch_sample_count = 0
        self.labeled_samples_dict: dict[str, str] = defaultdict(list)
        self.unlabeled_result_list: list[Tuple[float, ResultProvider]] = []
        self.threshold_samples_for_clustering: int = 1000 # cluster every N samples received from model
        self.max_unlabeled_for_clustering = 100 # max number of unlabeled used for clustering every cluster iteration
        self.num_selected_samples_per_iter: int = 5 ## number of sample to send to home after each clustering
        self.unlabeled_selection_mode: str = "top" ## or random
        self.n_clusters = 5
        self.random_state = 42
        self.semi_supervised = semi_supervised ## True if wanting to leverage existing labels as separate cluster groups, False for normal KMeans for only unlabeled samples
        self.unlabeled_list_lock = threading.Lock()

        threading.Thread(target=self.clustering_thread).start()
        threading.Thread(target=self.receive_labels_thread).start()
        

    def novel_clustering_process(self) -> None:
        logger.info("Running novel class discovery...")
        self.unlabeled_sample_count = 0
        
        while True:   
            ## this loop manages the incoming unlabeled samples from the queue and releases the clustering thread when n new samples have been inferenced.              

            with self.unlabeled_list_lock: ## ensure no new unlabeled samples can be retrieved while the current list of unlabeled samples is being processed in the clustering thread.  As soon as the list of samples for clustering is set, release other lock.
                result: ResultProvider = self.inbound_result_queue.get() ## grab each result provider object
                self.unlabeled_result_list.append((-result.score, result))
                self.batch_sample_count += 1
            
            
                ## if condition is met, set the clustering event.            
                if self.batch_sample_count == self.threshold_samples_for_clustering: 
                    self.cluster_iteration.set() ## trigger clustering
                    self.batch_sample_count = 0 ## reset batch counter
            

    def clustering_thread(self) -> None: 
                
        while True:
            logger.info("Running periodic clustering thread...\n")
            self.cluster_iteration.wait()            
            ## select unlabeled samples
            with self.unlabeled_list_lock: ## execute this block before any new unlabeled samples can be added to the current list, then reset the list
                ## Before sorting and slicing or randomly selecting, make sure to check if any of these unlabeled sample have been converted to labeled samples (i.e. they have been labeled in the time since they were received by this queue)
                self.unlabeled_result_list = sorted(self.unlabeled_result_list, key=lambda x: x[0]) ## sort the current state of the unlabeled results list
                if self.unlabeled_selection_mode == "top":
                    unlabeled_set_for_clustering: list[Tuple[float, ResultProvider]] = self.unlabeled_result_list[:self.max_unlabeled_for_clustering]  ## get the top N highest-scoring sample              
                else:
                    unlabeled_set_for_clustering: list[Tuple[float, ResultProvider]] = random.sample(self.unlabeled_result_list, min(len(self.unlabeled_result_list), self.max_unlabeled_for_clustering))
                self.unlabeled_result_list = []
            ## either sort the unlabeled list or pick randomly, configurable choice.  choosing the top 300 for example should find samples similar to the target class while random should find a wide variety.
            
            self.unlabeled_list_by_id: dict = {unlabeled_sample[1].id : unlabeled_sample[1] for unlabeled_sample in unlabeled_set_for_clustering}  ## dict with id as key and feature vector as value for all unlabeled samples.
            unlabeled_result_obj: list[ResultProvider] = list(self.unlabeled_list_by_id.values())
            
            unlabeled_feature_vectors: np.array = torch.stack([torch.load(io.BytesIO(result.feature_vector)) for result in unlabeled_result_obj]).numpy() ## numpy stack of vectors for clustering
                        
            ## choose whether to perform default kmeans or semi-supervised kmeans
            if not self.semi_supervised:
                sklearn_kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=self.random_state, n_init=1)
                sklearn_kmeans.fit(unlabeled_feature_vectors)  ## All default KMeans calls.
                sklearn_labels = sklearn_kmeans.labels_
                sklearn_centroids = sklearn_kmeans.cluster_centers_
            else:
                ## semi-supervised COP kmeans integration of labeled samples for clustering, next step.
                ## get labeled vectors
                ## separate function for semi-supervised.  Need to write must-link and cannot-link constraints.
                '''
                label_dirs = os.listdir(self.fv_dir)
                logger.info(f"Label dirs: {label_dirs}")
                for label_dir in list(self.labeled_samples_dict.keys()): ## loop through all dirs with labels
                '''
                
            
            ## After clustering of either type, choose which samples should be tiled and sent to home, e.g. 2 samples from each cluster
            selected_sample_indices = self.select_sample_labels(sklearn_labels)
            
            ## Final step: send selected tiles to home.  Future: write clustering results to scout for each iter.
            self.send_samples(list(operator.itemgetter(*selected_sample_indices)(unlabeled_result_obj)))           
            
            self.cluster_iteration.clear()
            
    def send_samples(self, selected_samples: list[ResultProvider]):
        ## For each selected sample, create tile and send to home        
        for selected_sample in selected_samples:            
            bboxes = [
                    BoundingBox(
                        x=bbox.get("x", 0.5),
                        y=bbox.get("y", 0.5),
                        w=bbox.get("w", 1.0),
                        h=bbox.get("h", 1.0),
                        class_name=class_name_to_str(bbox["class_name"]),
                        confidence=bbox["confidence"],
                    )
                    for bbox in selected_sample.bboxes
                ]
        
            tile = SendTile(
                    objectId=selected_sample.id,
                    scoutIndex=self.scout_index,
                    version=selected_sample.model_version, 
                    feature_vector=selected_sample.feature_vector,
                    attributes=selected_sample.attributes.get(),
                    boundingBoxes=bboxes,
                    novel_sample=True,
                )
            
            self.outbound_pipe.send(tile.SerializeToString()) ## send each sample to home
            #logger.info(f" Sent item {i} to home...\n")
                        

    def receive_labels_thread(self) -> None:
        while True:
            logger.info("Waiting in label thread...")
            label, labeled_fv_path = self.labels_queue.get()
            self.labeled_samples_dict[label].append(labeled_fv_path) ## append the paths of each feature vector according to the dict according to their label.  Don't actually load until ready to cluster.          
            #logger.info(f"Got a label: {label_path}")
            
    
    def select_sample_labels(self, labels:list, num_samples_per_label:int=2) -> list:
        """
        Randomly samples a specified number of indices for each unique label in a list.
        Args:
            labels: A list of labels.
            num_samples_per_label: The number of samples to randomly select for each label.
        Returns:
            A list of indices of the sampled elements. Returns an empty list if there are fewer samples than requested for a label.
        """
        unique_labels = np.unique(labels)
        sampled_indices = []

        for label in unique_labels:
            label_indices = np.where(labels == label)[0]  # Get indices of elements with the current label
            num_available = len(label_indices)

            if num_available == 0:
                print(f"Warning: Label {label} has no samples.")
                continue  # Skip if no samples for this label

            num_to_sample = min(num_samples_per_label, num_available)  # Sample all available if fewer than requested

            rng = np.random.RandomState() #Create a local random number generator to avoid side effects
            rng.shuffle(label_indices)  # Shuffle the indices randomly
            sampled_indices.extend(label_indices[:num_to_sample])  # Select the appropriate number of indices

        return sampled_indices

def main(result_queue: mp.Queue, labels_queue: mp.Queue, home_pipe: _ConnectionBase, clustering_dir: Path, scout_index: int):
    ### create clustering object
    novel_clustering = Novel_Class_Clustering(result_queue, labels_queue, home_pipe, clustering_dir, scout_index)
    novel_clustering.novel_clustering_process()


