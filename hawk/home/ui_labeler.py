# SPDX-FileCopyrightText: 2022,2023 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import argparse
import csv
import glob
import imghdr
import json
import multiprocessing as mp
import os
import queue
import time
from os import walk
from pathlib import Path

import numpy as np
from flask import (
    Flask,
    Response,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from logzero import logger


class EndpointAction(object):

    def __init__(self, action):
        self.action = action
        self.response = Response(status=200, headers={})

    def __call__(self, *args, **kwargs):
        # Perform the action
        response = self.action(*args, **kwargs)
        if response != None:
            return response
        else:
            return self.response
    
class UILabeler(object):
    app = None

    def __init__(self, mission_dir, save_automatically=False):
        self.app = Flask(__name__)
        self.app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

        mission_dir = str(mission_dir)
        self._image_dir = os.path.join(mission_dir, 'images')
        self._meta_dir = os.path.join(mission_dir, 'meta')
        self._label_dir = os.path.join(mission_dir, 'labels')
        self._stats_dir = os.path.join(mission_dir, 'logs')
        self._label_map = {
            '-1': {'color': '#ffffff', 
                    'text': 'UNLABELED'},
            '0': {'color': '#ae163e',
                   'text': 'NEGATIVE'},
            '1': {'color': 'green',
                  'text': 'POSITIVE'}
        }
        self._stats_keys = {
            'version' : "Model Version : ",
            'totalObjects': "Total Tiles : ", 
            'processedObjects': "Tiles Processed : ",
            'positives': "Positives Labeled : ",
            'negatives': "Negatives Labeled : "}
        
        directory = self._image_dir
        if directory[len(directory) - 1] != "/":
             directory += "/"
        self.app.config["IMAGES"] = directory
        self.app.config["LABELS"] = []
        self.app.config["HEAD"] = 0
        self.app.config["LOGS"] = self._stats_dir
        self.label_changed = False
        self.save_auto = save_automatically
         
        files = None
        for (_, _, filenames) in walk(self.app.config["IMAGES"]):
            files = sorted(filenames)
            break
        if files == None:
            logger.error("No files")
            exit()
        else:
            self.app.config["LABELS"] = ['-1'] * len(files)
        self.app.config["FILES"] = files
        self.not_end = True 
        
        self.image_boxes = []
        self.num_thumbnails = 4
        self.add_all_endpoints()

    def start_labeling(self, input_q, result_q, stats_q, stop_event):
        self.input_q = input_q
        self.result_q = result_q
        self.stop_event = stop_event
        self.stats_q = stats_q
        self.positives = 0 
        self.negatives = 0 
        self.bytes = 0

        try:
            self.app.run(port=8000, use_reloader=False)
        except KeyboardInterrupt as e:
            raise e 
        
    def run(self):
        self.result_q = mp.Queue()
        self.stats_q = mp.Queue()
        self.positives = 0 
        self.negatives = 0 
        self.bytes = 0
        self.app.jinja_env.filters['bool'] = bool
        self.app.run(port=8000)

        
    def add_all_endpoints(self):
        # Add endpoints
        self.add_endpoint(endpoint='/', endpoint_name='index', handler=self.index)
        self.add_endpoint(endpoint='/next', endpoint_name='next', handler=self.next)
        self.add_endpoint(endpoint='/prev', endpoint_name='prev', handler=self.prev)
        self.add_endpoint(endpoint='/backward', endpoint_name='backward', handler=self.backward)
        self.add_endpoint(endpoint='/forward', endpoint_name='forward', handler=self.forward)
        self.add_endpoint(endpoint='/save', endpoint_name='save', handler=self.save)
        self.add_endpoint(endpoint='/undo', endpoint_name='undo', handler=self.undo)
        self.add_endpoint(endpoint='/add/<id>', endpoint_name='add', handler=self.add)
        self.add_endpoint(endpoint='/remove/<id>', endpoint_name='remove', handler=self.remove)
        self.add_endpoint(endpoint='/image/<f>', endpoint_name='images', handler=self.images)
        self.add_endpoint(endpoint='/classify/<id>', endpoint_name='classify', handler=self.classify)
        
    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None):
        self.app.add_url_rule(endpoint, endpoint_name, EndpointAction(handler)) 
        # You can also add options here : "... , methods=['POST'], ... "

    # ==================== ------ API Calls ------- ====================
    def index(self, *args, **kwargs):
        self.reload_directory()
        if (len(self.app.config["FILES"]) == 0):
            self.not_end = False
            self.app.config["HEAD"] = -1 
            return render_template('index.html', 
                                   directory="", images=[""], 
                                   head=0, 
                                   len=len(self.app.config["FILES"]),
                                   color='#ffffff', label_text="", 
                                   boxes=self.image_boxes,
                                   stats={},
                                   label_changed= 1 if self.label_changed else 0,
            )
        
        directory = self.app.config['IMAGES']

        index_num = self.app.config["HEAD"]
        length_files = len(self.app.config["FILES"])
        
        # main image and thumbnails
        image_paths = []
        for i in range(self.num_thumbnails+1):
            idx = index_num + i
            if idx >= length_files:
                break
            image_paths.append(self.app.config["FILES"][idx])

        if (self.app.config["HEAD"] == 0 and not self.label_changed):
            label, boxes = self.read_labels()
            self.app.config["LABELS"][self.app.config["HEAD"]] = label
            self.image_boxes = boxes
        else:
            label = self.app.config["LABELS"][self.app.config["HEAD"]]
            
        color = self._label_map[label]['color']
        label_text = self._label_map[label]['text']
        self.not_end = not(self.app.config["HEAD"] == length_files - 1)
        condition_changed = self.label_changed and not self.save_auto
        search_stats = self.get_latest_stats()

        return render_template('index.html', 
                               directory=directory, images=image_paths, 
                               head=self.app.config["HEAD"] + 1, 
                               files=len(self.app.config["FILES"]),
                               color=color, label_text=label_text,
                               boxes=self.image_boxes,
                               stats=search_stats,
                               label_changed=int(condition_changed == True),
                               save=int(self.save_auto == True))

    def reload_directory(self):
        old_length = len(self.app.config["FILES"])
        for (dirpath, dirnames, filenames) in walk(self.app.config["IMAGES"]):
            files = sorted(filenames)
            break
        if files == None:
            logger.error("No files")
            exit()
        self.app.config["FILES"] = files
        new_files = len(self.app.config["FILES"]) - old_length
        [self.app.config["LABELS"].append('-1') for i in range(new_files)]
        self.not_end = not(self.app.config["HEAD"] == len(self.app.config["FILES"]) - 1)
        if new_files and self.app.config["HEAD"] < 0:
            self.app.config["HEAD"] = 0
        return

    def read_labels(self, head_id=-1):
        if head_id == -1:
            head_id = self.app.config["HEAD"]
        path = self.app.config["FILES"][head_id]
        data_name = os.path.splitext(path)[0]
        label_path = os.path.join(self._label_dir, f"{data_name}.json")

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                image_labels = json.load(f)
                
            label = str(image_labels['imageLabel'])
            assert label in ['1', '0'], f"Unkonown label {label}"
            image_boxes = image_labels['boundingBoxes']
            boxes = []
            for i, box in enumerate(image_boxes):
                _, xMin, xMax, yMin, yMax = box.split(" ")
                id = i + 1
                boxes.append({"id":id, 
                              "xMin":xMin, 
                              "xMax":xMax, 
                              "yMin":yMin, 
                              "yMax":yMax})
            
            return label, boxes
        
        return '-1', [] 
       
    
    def write_labels(self): 
        if not self.label_changed:
            return 
        path = self.app.config["FILES"][self.app.config["HEAD"]]
        data_name = os.path.splitext(path)[0]
        meta_path = os.path.join(self._meta_dir, f"{data_name}.json")
        label_path = os.path.join(self._label_dir, f"{data_name}.json")
        image_label = self.app.config["LABELS"][self.app.config["HEAD"]] 

        if image_label == '-1':
            if os.path.exists(label_path):
                os.remove(label_path)

        if image_label not in ['1', '0']:
            return

        data = {}
        with open(meta_path, "r") as f:
            data = json.load(f)
        self.bytes += data['size']
        # assuming binary class hardcoding positive id as 0
        bounding_boxes = [ ]
        
        for box in self.image_boxes:
            # convert to yolo fomat
            bounding_boxes.append(" ".join(['0', 
                                            box['xMin'],
                                            box['xMax'],
                                            box['yMin'],
                                            box['yMax']])
            )
        label = {
            'objectId': data['objectId'],
            'scoutIndex': data['scoutIndex'],
            'imageLabel': image_label, 
            'boundingBoxes': bounding_boxes, 
        }    
        with open(label_path, "w") as f:
            json.dump(label, f, indent=4, sort_keys=True)
             
        self.result_q.put(label_path)
        image_labels = np.array(self.app.config["LABELS"])
        self.positives = len(np.where(image_labels == '1')[0])
        self.negatives = len(np.where(image_labels == '0')[0])
        logger.info("({}, {}) Labeled {}".format(self.positives, self.negatives, data['objectId']))
        self.stats_q.put((self.positives, self.negatives, self.bytes))
        self.label_changed = False
        return redirect(url_for('index'))

    def save(self, *args, **kwargs):
        self.write_labels()
        return redirect(url_for('index'))

    def undo(self, *args, **kwargs):
        self.label_changed = True
        label = '-1'
        self.app.config["LABELS"][self.app.config["HEAD"]] = label
        if label != '1':
            self.image_boxes = []
        self.write_labels()
        self.label_changed = False
        return redirect(url_for('index'))

    def next(self, *args, **kwargs):
        if self.save_auto:
            self.write_labels()
        if self.not_end:
            self.app.config["HEAD"] = self.app.config["HEAD"] + 1
        else:
            logger.info("Waiting for Results ...")
            while (not self.not_end):
                time.sleep(1)
                self.reload_directory()

        label, boxes = self.read_labels()
        self.app.config["LABELS"][self.app.config["HEAD"]] = label
        self.image_boxes = boxes
        self.label_changed = False
        return redirect(url_for('index'))

    def backward(self, *args, **kwargs):
        if self.save_auto:
            self.write_labels()

        found_positive = False
        curr_id = self.app.config["HEAD"]
        new_id = curr_id
        length_files = len(self.app.config["FILES"])

        while (not found_positive and new_id - 1 >= 0):
            new_id -= 1
            label, boxes = self.read_labels(new_id)
            if label == '1':
                found_positive = True
            
        if found_positive:
            self.app.config["HEAD"] = new_id
        
        label, boxes = self.read_labels()
        self.app.config["LABELS"][self.app.config["HEAD"]] = label
        self.image_boxes = boxes
        self.label_changed = False

        return redirect(url_for('index'))

    def forward(self, *args, **kwargs):
        if self.save_auto:
            self.write_labels()

        found_positive = False
        curr_id = self.app.config["HEAD"]
        new_id = curr_id
        length_files = len(self.app.config["FILES"])

        while (not found_positive and new_id + 1 < length_files):
            new_id += 1
            label, boxes = self.read_labels(new_id)
            if label == '1':
                found_positive = True
            
        if found_positive:
            self.app.config["HEAD"] = new_id
        
        label, boxes = self.read_labels()
        self.app.config["LABELS"][self.app.config["HEAD"]] = label
        self.image_boxes = boxes
        self.label_changed = False

        return redirect(url_for('index'))

    def prev(self, *args, **kwargs):
        if (self.app.config["HEAD"] == 0):
            return redirect(url_for('index'))
        self.app.config["HEAD"] = self.app.config["HEAD"] - 1
        label, boxes = self.read_labels()
        self.app.config["LABELS"][self.app.config["HEAD"]] = label
        self.image_boxes = boxes
        self.label_changed = False
        return redirect(url_for('index'))

    def add(self, *args, **kwargs):
        id = kwargs['id']
        self.label_changed = True
        self.app.config["LABELS"][self.app.config["HEAD"]] = '1'
        xMin = request.args.get("xMin")
        xMax = request.args.get("xMax")
        yMin = request.args.get("yMin")
        yMax = request.args.get("yMax")
        self.image_boxes.append({"id":id, 
                                    "xMin":xMin, 
                                    "xMax":xMax, 
                                    "yMin":yMin, 
                                    "yMax":yMax})
        return redirect(url_for('index'))   

    def remove(self, *args, **kwargs):
        self.label_changed = True
        id = kwargs['id']
        index = int(id) - 1
        del self.image_boxes[index]
        for label in self.image_boxes[index:]:
            label["id"] = str(int(label["id"]) - 1)
        if not len(self.image_boxes):
            self.app.config["LABELS"][self.app.config["HEAD"]] = '-1'
        return redirect(url_for('index'))

    def images(self, *args, **kwargs):
        f = kwargs['f']
        images = self.app.config['IMAGES']
        return send_file(images + f)

    def classify(self, *args, **kwargs):
        self.label_changed = True
        label = request.args.get("label")
        self.app.config["LABELS"][self.app.config["HEAD"]] = label
        if label != '1':
            self.image_boxes = []
        if label == '-1':
            self.undo()
        
        return redirect(url_for('index'))

    def get_latest_stats(self):
        list_of_files = glob.glob(self._stats_dir+'/stats-*') 
        latest_stats = max(list_of_files, key=os.path.getctime)
        search_stats = {}
        with open(latest_stats, "r") as f:
            search_stats = json.load(f)
        stats = []
        for k, v in self._stats_keys.items():
            stats_item = search_stats.get(k,0)
            if k == "positives":
                stats_item = max(stats_item, self.positives)
            elif k == "negatives":
                stats_item = max(stats_item, self.negatives)
            stats_text = v + str(int(stats_item))
            stats.append(stats_text)
        return stats

    
# mission = "/home/shilpag/Documents/label_ui/dota-20-30k_swimming-pool_30_20221031-121302"
# mission = "/home/eric/School_Bus_Hawk/Hawk_Mission_Data/test-hawk-school_bus_tiles_video_20221010-131323"
# a = UILabeler(mission)
# a.run()
# python ui_labeler.py
