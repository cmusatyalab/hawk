import argparse
import csv
import json 
import queue
import imghdr
import os
import time
import multiprocessing as mp
import numpy as np

from os import walk
from flask import Flask, Response, redirect, url_for, request
from flask import render_template
from flask import send_file
from logzero import logger
from pathlib import Path

from home import *


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
        self._label_map = {
            '-1': {'color': '#ffffff', 
                    'text': 'UNLABELED'},
            '0': {'color': '#ae163e',
                   'text': 'NEGATIVE'},
            '1': {'color': 'green',
                  'text': 'POSITIVE'}
        }
        
        directory = self._image_dir
        if directory[len(directory) - 1] != "/":
             directory += "/"
        self.app.config["IMAGES"] = directory
        self.app.config["LABELS"] = []
        self.app.config["HEAD"] = 0
        self.label_changed = False
        self.save_auto = save_automatically
         
        files = None
        for (_, _, filenames) in walk(self.app.config["IMAGES"]):
            files = sorted(filenames)
            break
        if files == None:
            print("No files")
            exit()
        else:
            self.app.config["LABELS"] = ['-1'] * len(files)
        self.app.config["FILES"] = files
        self.not_end = True 
        
        self.image_boxes = []
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
                                   directory="", image="", 
                                   head=0, 
                                   len=len(self.app.config["FILES"]),
                                   color='#ffffff', label_text="", 
                                   boxes=self.image_boxes,
                                   label_changed= 1 if self.label_changed else 0)
        
        directory = self.app.config['IMAGES']
        image_path = self.app.config["FILES"][self.app.config["HEAD"]]

        if (self.app.config["HEAD"] == 0 and not self.label_changed):
            label, boxes = self.read_labels()
            self.app.config["LABELS"][self.app.config["HEAD"]] = label
            self.image_boxes = boxes
        else:
            label = self.app.config["LABELS"][self.app.config["HEAD"]]
            
        color = self._label_map[label]['color']
        label_text = self._label_map[label]['text']
        self.not_end = not(self.app.config["HEAD"] == len(self.app.config["FILES"]) - 1)
        condition_changed = self.label_changed and not self.save_auto

        return render_template('index.html', 
                               directory=directory, image=image_path, 
                               head=self.app.config["HEAD"] + 1, 
                               files=len(self.app.config["FILES"]),
                               color=color, label_text=label_text,
                               boxes=self.image_boxes,
                               label_changed=int(condition_changed == True),
                               save=int(self.save_auto == True))

    def reload_directory(self):
        old_length = len(self.app.config["FILES"])
        for (dirpath, dirnames, filenames) in walk(self.app.config["IMAGES"]):
            files = sorted(filenames)
            break
        if files == None:
            print("No files")
            exit()
        self.app.config["FILES"] = files
        new_files = len(self.app.config["FILES"]) - old_length
        [self.app.config["LABELS"].append('-1') for i in range(new_files)]
        self.not_end = not(self.app.config["HEAD"] == len(self.app.config["FILES"]) - 1)
        if new_files and self.app.config["HEAD"] < 0:
            self.app.config["HEAD"] = 0
        return

    def read_labels(self):
        path = self.app.config["FILES"][self.app.config["HEAD"]]
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
        print(self.image_boxes)
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
    
# mission = "/home/shilpag/Documents/hawk-data/dota-tennis_20220919-080839"
# a = UILabeler(mission)
# a.run()
