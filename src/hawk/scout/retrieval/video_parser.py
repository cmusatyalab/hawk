# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import time

import cv2
import numpy as np
from logzero import logger

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
    logger.info(type(capture))
    status, frame = capture.read()
    logger.info(status)
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
                time.sleep(10) # artificial delay
            except Exception as e:
                logger.info(e)
        status, frame = capture.read()


