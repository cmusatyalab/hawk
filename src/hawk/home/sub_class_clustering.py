# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import json
import queue
import threading
import time
import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import torch
from pathlib import Path
import threading, sys
from logzero import logger
from sklearn.cluster import KMeans
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple
from .label_utils import ClassMap, Detection, LabelSample, read_jsonl


class SubClassClustering:
    def __init__(self, mission_dir: Path, cluster_sample_interval: int = 1):
        self.mission_dir = mission_dir
        self.cluster_sample_interval = cluster_sample_interval
        self.labeled_json_file = mission_dir / "labeled.jsonl"
        self.feature_vector_list: Sequence[Path] = []
        self.entry_list: Sequence[LabelSample] = []
        self.cluster_iteration = threading.Event()
        self.sub_class_label_file = self.mission_dir / "sub_class_labels.jsonl"
        threading.Thread(target=self.clustering_thread).start()      


    def sub_class_clustering_process(self):
        print(f"Starting clustering process...")
        samples_since_last_cluster = 0
        total_samples = 0
        self.feature_vector_stack: torch.Tensor | None = None ## stack of feature vectors that grows over time during the mission.        
        for new_entry in read_jsonl(self.labeled_json_file, tail=True):
            if not new_entry.detections: ## skip if not a positive
                continue
            self.entry_list.append(new_entry)            
            fv_path = new_entry.unique_name(self.mission_dir / "feature_vectors", ".pt")
            new_vector = torch.load(fv_path).unsqueeze(0)
            self.feature_vector_list.append(fv_path)
            if self.feature_vector_stack is None: ## initialize feature vector stack
                self.feature_vector_stack = new_vector
            else:
                self.feature_vector_stack = torch.cat((self.feature_vector_stack, new_vector), dim=0) ## concatenate next feature vector
                samples_since_last_cluster += 1            
            total_samples += 1

            if (samples_since_last_cluster == self.cluster_sample_interval) and (total_samples > 1): ## start if more than 1 vector and sufficient number of fvs
                self.cluster_iteration.set() ## allow clustering process to execute on current set of feature vectors                
                samples_since_last_cluster = 0 ## reset counter
            
            ## will also need to add functionality here that allows a button press from the browser to trigger the clustering thread loop.


    def clustering_thread(self):
        ### thread that waits for sufficient number of new samples to execute clustering algorithm        
        print("Starting sub class clustering thread...")
        while True:
            self.cluster_iteration.wait() ## wait until sufficient number of new labeled positives have been added to batch of feature vectors
            kmeans = KMeans(n_clusters=2, n_init=10)
            np_tensors = self.feature_vector_stack.numpy()
            kmeans.fit(np_tensors)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            
            ## write sub-class labels into each entry into jsonl file
            logger.info("Writing cluster sub-class labels to file...")
            with open(self.sub_class_label_file, 'w') as fp:
                for i, label in enumerate(labels):
                    self.entry_list[i].to_jsonl(fp, sub_class=str(label))
            self.cluster_iteration.clear()

def main(mission_dir) -> int:    
    clustering = SubClassClustering(mission_dir) 
    clustering.sub_class_clustering_process()
    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recluster_sample_interval", type=int, default=1)
    parser.add_argument("--mission_directory", type=Path, nargs="?", default=".")
    args = parser.parse_args()

    clustering = SubClassClustering(args.mission_directory)
    clustering.sub_class_clustering_process()




