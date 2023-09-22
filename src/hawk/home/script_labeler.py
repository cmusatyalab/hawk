# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import json
import queue
import time
from pathlib import Path
from typing import Dict

from logzero import logger


class ScriptLabeler:
    def __init__(self,
                 label_dir: Path,
                 configuration: Dict,
                 gt_path: str = "",
                 label_mode: str = "classify") -> None:

        self._label_dir = label_dir
        self._gt_path = Path(gt_path)
        self.ordered_queue = queue.PriorityQueue()
        self._token = False
        self._label_time = 0
        # Token selector code to modify labeling process.
        self.configuration = configuration
        selector_field = self.configuration['selector']
        if selector_field['type'] == 'token':
            self._token = True
            init_samples = selector_field['token']['initial_samples']
            num_scouts = len(self.configuration['scouts'])
            self.total_init_samples = int(init_samples * int(num_scouts))
            self._label_time = float(selector_field['token']['label_time'])
        ##########

        if label_mode == "classify":
            self.labeling_func = self.classify
        elif label_mode == "detect":
            self.labeling_func = self.detect
        else:
            raise NotImplementedError(f"Labeling mode not known {label_mode}")


    def start_labeling(self, input_q, result_q, stats_q, stop_event):
        self.input_q = input_q
        self.result_q = result_q
        self.stop_event = stop_event
        self.stats_q = stats_q
        self.positives = 0
        self.negatives = 0
        self.bytes = 0
        self.received_samples = 0


        try:
            self.labeling_func()
        except KeyboardInterrupt as e:
            raise e

    def classify(self):

        # Object ID contains label
        # if /1/ in Id then label = 1 else 0
        try:
            while not self.stop_event.is_set():
                try:
                    meta_path = self.input_q.get() ## get the meta path for the next sample to label
                    self.received_samples += 1
                except queue.Empty:
                    continue

                data_name = meta_path.name
                logger.info(data_name)

                label_path = self._label_dir / f"{data_name}"

                data = {}
                with open(meta_path, "r") as f: ## get the data from the meta_data file
                    data = json.load(f)

                image_label = '1' if '/1/' in data['objectId'] else '0'

                if image_label == '1':
                    self.positives += 1
                else:
                    self.negatives += 1
                time.sleep(self._label_time)

                self.bytes += data['size']

                label = {
                    'objectId': data['objectId'],
                    'scoutIndex': data['scoutIndex'],
                    'imageLabel': image_label,
                    'boundingBoxes': [],
                }

                with open(label_path, "w") as f:
                    json.dump(label, f)

                self.result_q.put(label_path)
                logger.info("({}, {}) Labeled {}".format(self.positives, self.negatives, data['objectId']))
                self.stats_q.put((self.positives, self.negatives, self.bytes))
        except (IOError, KeyboardInterrupt) as e:
            logger.error(e)


    def detect(self):

        assert self._gt_path.exists(), "GT Dir does not exist"
        # Takes labels from file: _gt_path/<basename>.txt
        try:
            while not self.stop_event.is_set():
                try:
                    meta_path = self.input_q.get()
                except queue.Empty:
                    continue

                data_name = meta_path.name

                logger.info(data_name)


                label_path = self._label_dir / f"{data_name}"

                data = {}
                with open(meta_path, "r") as f:
                    data = json.load(f)

                basename = Path(data['objectId']).name
                label_file = self._gt_path / (basename.split('.')[0]+".txt")

                bounding_boxes = []
                if label_file.exists():
                    bounding_boxes = open(label_file).read().splitlines()

                if len(bounding_boxes):
                    image_label = '1'
                    self.positives += 1
                else:
                    image_label = '0'
                    self.negatives += 1

                self.bytes += data['size']

                label = {
                    'objectId': data['objectId'],
                    'scoutIndex': data['scoutIndex'],
                    'imageLabel': image_label,
                    'boundingBoxes': bounding_boxes,
                }

                with open(label_path, "w") as f:
                    json.dump(label, f)

                self.result_q.put(label_path)
                logger.info("({}, {}) Labeled {}".format(self.positives, self.negatives, data['objectId']))
                self.stats_q.put((self.positives, self.negatives, self.bytes))
        except (IOError, KeyboardInterrupt) as e:
            logger.error(e)




