# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import copy
import gc
import cv2
import io
import queue
import os, shutil, subprocess
import random
import numpy as np 
import time
import threading
import multiprocessing as mp
from typing import Iterable, Sized
from collections import defaultdict
from PIL import Image
from logzero import logger

from hawk.core.object_provider import ObjectProvider
from hawk.proto.messages_pb2 import HawkObject, FileDataset, Streaming_Video
from hawk.retrieval.diamond_attribute_provider import DiamondAttributeProvider
from hawk.retrieval.retriever import Retriever
from hawk.retrieval.retriever_stats import RetrieverStats
from hawk.core.utils import get_server_ids
from hawk.core.utils import ATTR_GT_LABEL
'''
class VideoFrameProducer:
    def __init__(self, video_source):
        self.video_source = video_source
        logger.info("About to load the video file...")
        self.capture = cv2.VideoCapture(self.video_source)
        logger.info("Finished loading video file...")
'''

def produce_video_frames(producer_queue, video_source):
    logger.info("About to load the video file...")
    capture = cv2.VideoCapture(video_source)
    logger.info("Finished loading video file...")
    status, frame = capture.read()
    logger.info("Pushed first frame into queue...")
    frame_num = 0
    sample_num = 0
    while status:
        frame_num += 1
        if frame_num % 15 == 0: ## add fps functionality later
            sample_num += 1
            try:
                frame = np.array(frame)
                producer_queue.put(("scout_1_" + str(sample_num) + ".jpeg", frame))
                logger.info(f"Put frame {sample_num} in the queue...")
                time.sleep(5) # artificial delay
            except Exception as e:
                logger.info(e)
        status, frame = capture.read()

        
