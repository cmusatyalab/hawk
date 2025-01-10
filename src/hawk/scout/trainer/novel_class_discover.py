# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import time
import queue
from pathlib import Path
from collections import defaultdict
import torch, io
from logzero import logger
import multiprocessing as mp
from multiprocessing.connection import _ConnectionBase
import threading
from ...classes import ClassCounter, ClassName, class_name_to_str
from ...proto.messages_pb2 import (
    BoundingBox,
    SendTile
)


class Novel_Class_Clustering:
    def __init__(self, inbound_result_queue: mp.Queue, labels_queue: mp.Queue, outbound_pipe: _ConnectionBase, clustering_dir: Path, scout_index: int, cluster_interval: int = 1000):
        self.clustering_dir = clustering_dir
        self.inbound_result_queue = inbound_result_queue
        self.labels_queue = labels_queue
        self.feature_vector = None
        self.scout_index = scout_index
        self.cluster_iteration = threading.Event()
        self.outbound_pipe = outbound_pipe
        self.batch_sample_count = 0
        self.labeled_samples_dict = defaultdict(list)
        self.unlabeled_result_queue = queue.PriorityQueue()
        self.threshold_samples_for_clustering = 500 # cluster every N samples received from model
        self.num_selected_samples_per_iter = 5 ## number of sample to send to home after each clustering

        threading.Thread(target=self.clustering_thread).start()
        threading.Thread(target=self.receive_labels_thread).start()
        

    def novel_clustering_process(self) -> None:
        logger.info("Running novel class discovery...")
        new_samples = {}
        tiles_to_send = []
        max_num_unknown_samples = 300
        self.unlabeled_sample_count = 0
        
        while True:   
            ## this loop manages the incoming unlabeled samples from the queue and releases the clustering thread when n new samples have been inferenced.         
            ## need a dict sorted on score           

            result = self.inbound_result_queue.get() ## grab each result provider object
            self.unlabeled_result_queue.put((-result.score, result))
            self.batch_sample_count += 1
            
            ## if condition is met, set the clustering event.            
            if self.batch_sample_count == self.threshold_samples_for_clustering: 
                self.cluster_iteration.set() ## trigger clustering
                self.batch_sample_count = 0 ## reset batch counter
            
            ## extract list from unlabeled queue once clustering has been triggered.
            

    def clustering_thread(self) -> None: 
                
        while True:
            logger.info("Running periodic clustering thread...\n")
            selected_samples = []
            tiles_to_send = []
            self.cluster_iteration.wait()
            ## need some concurrency control here.
            
            ## combine the labeled samples with the unlabeled samples

            ## load all labeled samples into torch tensor
            
            ## read labeled samples from mission dir
            ## negative labeled samples feature vectors are saved to feature_vectors/
            ## positives labeld sampels fv are saved to mission dir

            ## vector = torch.load(io.BytesIO(result.feature_vector))
            ## end of each iteration should generate handful of samples to send to home, write files for results of cluster iteration.
            time.sleep(5)
            ## clustering complete
            ## select set of samples to transmit to home and create tiles and send over pipe
            for _ in range(self.num_selected_samples_per_iter):
                temp_sample = self.unlabeled_result_queue.get() ## temporary sample selection for testing.
                selected_samples.append(temp_sample[1])
                        
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
                tiles_to_send.append(tile)
            for t in tiles_to_send:
                self.outbound_pipe.send(tile.SerializeToString()) ## send all samples to home
            logger.info(f"Just sent {len(tiles_to_send)} novel samples to home...")
            #time.sleep(5)

            self.cluster_iteration.clear()

    def receive_labels_thread(self) -> None:
        while True:
            logger.info("Waiting in label thread...")
            label, label_path = self.labels_queue.get()
            self.labeled_samples_dict[label].append(label_path) ## append the paths of each feature vector according to the dict according to their label.  Don't actually load until ready to cluster.          
            #logger.info(f"Got a label: {label_path}")
            
                    

def main(result_queue: mp.Queue, labels_queue: mp.Queue, home_pipe: _ConnectionBase, clustering_dir: Path, scout_index: int):
    ### create clustering object
    novel_clustering = Novel_Class_Clustering(result_queue, labels_queue, home_pipe, clustering_dir, scout_index)
    novel_clustering.novel_clustering_process()


'''
Need to keep track of all samples that have a valid label returned.  This set of samples should be automatically added to the total tensor for clustering. For a typical mission, it might be around 200 or so (if 1400-1600 total labels).  Then we take the top N scores of samples in the queue.

A separate approach could be to do a random sampling of all unlabeled samples regardless of score.  We can then compare performance for these two approaches.


Need to answer this question:
1) Should we cluster on all positives across all scouts? Yes: When transmitting every sample to home, save feature vector on respective scout.  Then when sending positive label to other scouts, make sure to send feature vector with it.
If so, we need to find a way to transfer the feature vectors during hte s2s process (s2s_add_tile_and_label) in data manager where the image and labeltile is sent to all the other scouts.  The only place it makes sense is to modify read_object in retriever.py and save to disk each fv for each tile upon initial inference... Or when sending the FV with the sample to home, we send it back with the FV in the SendTile() protobuf.  Then we would have to make sure it is added to the  

2) Should we cluster on local labeled positives only, local labeled negatives only, and local unlabeled?
How many local unlabeled should we use to cluster?  We should set some upper limit on total samples to cluster... or we can set a lower bound score on which to cluster... as in use all local unlabeled samples with score above 0.7.  Say we have 10 local (+) 50 (-) labeled, and 200 unlabeled local samples above 0.7.

ultimately it might just be easier to send the FV with each sample to home and send it back with the sendlabel protobuf.  that way we dont have to write every FV upon inference, which would be ~30k FVs for each mission.


During mission:
1) Get each inferenced sample one at a time with its FV, put into a priority queue according to score
2) Get each labeled senLabel object to keep track of positive labeled samples and negative labeled samples
As senlabel object come in from the queue, store them in a vector by class
3) Once reaching a certain condition

Select the cluster with the closest centroid to the positive class, (and not the labeled negatives).
Pick N samples from that cluster closest to its center.
'''
