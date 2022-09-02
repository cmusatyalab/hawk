import argparse
import csv
import json 
import queue
import imghdr
import os
import time
import multiprocessing as mp

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

    def __init__(self, mission_dir):
        self.app = Flask(__name__)
        self.app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

        mission_dir = str(mission_dir)
        self._image_dir = os.path.join(mission_dir, 'images')
        self._meta_dir = os.path.join(mission_dir, 'meta')
        self._label_dir = os.path.join(mission_dir, 'labels')
        
        directory = self._image_dir
        if directory[len(directory) - 1] != "/":
             directory += "/"
        self.app.config["IMAGES"] = directory
        self.app.config["LABELS"] = []
        self.app.config["HEAD"] = 0
        files = None
        for (dirpath, dirnames, filenames) in walk(self.app.config["IMAGES"]):
            files = sorted(filenames)
            break
        if files == None:
            print("No files")
            exit()
            self.app.config["HEAD"] = -1
        else:
            self.app.config["LABELS"] = ['-1'] * len(files)
        self.app.config["FILES"] = files
        
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
            self.app.run(use_reloader=False)
        except KeyboardInterrupt as e:
            raise e 
        
    def run(self):
        self.result_q = mp.Queue()
        self.stats_q = mp.Queue()
        self.positives = 0 
        self.negatives = 0 
        self.bytes = 0
        self.app.run()
        
    def add_all_endpoints(self):
        # Add root endpoint
        self.add_endpoint(endpoint='/', endpoint_name='index', handler=self.index)
        self.add_endpoint(endpoint='/next', endpoint_name='next', handler=self.next)
        self.add_endpoint(endpoint='/prev', endpoint_name='prev', handler=self.prev)
        self.add_endpoint(endpoint='/image/<f>', endpoint_name='images', handler=self.images)
        self.add_endpoint(endpoint='/classify/<id>', endpoint_name='classify', handler=self.classify)
        
    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None):
        self.app.add_url_rule(endpoint, endpoint_name, EndpointAction(handler)) 
        # You can also add options here : "... , methods=['POST'], ... "

    # ==================== ------ API Calls ------- ====================
    def index(self, *args, **kwargs) :
        label_map = {
            '-1': {'color': '#ffffff', 
                    'text': 'UNLABELED'},
            '0': {'color': '#ae163e',
                   'text': 'NEGATIVE'},
            '1': {'color': 'green',
                  'text': 'POSITIVE'}
        }
        self.reload_directory()
        if (len(self.app.config["FILES"]) == 0):
            self.app.config["HEAD"] = -1 
            return render_template('index.html', not_end=True, 
                                   directory="", image="", 
                                   head=0, 
                                   len=len(self.app.config["FILES"]),
                                   color='#ffffff', label_text="")
        
        if (self.app.config["HEAD"] == len(self.app.config["FILES"])):
            logger.info("Waiting for Results ...")
            while (self.app.config["HEAD"] == len(self.app.config["FILES"])):
                time.sleep(1)
                self.reload_directory()
        directory = self.app.config['IMAGES']
        image = self.app.config["FILES"][self.app.config["HEAD"]]
        label = self.app.config["LABELS"][self.app.config["HEAD"]]
        color = label_map[label]['color']
        label_text = label_map[label]['text']
        not_end = not(self.app.config["HEAD"] == len(self.app.config["FILES"]) - 1)
        return render_template('index.html', not_end=not_end, 
                               directory=directory, image=image, 
                               head=self.app.config["HEAD"] + 1, 
                               len=len(self.app.config["FILES"]),
                               color=color, label_text=label_text)

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
        return

    def next(self, *args, **kwargs):
        self.app.config["HEAD"] = self.app.config["HEAD"] + 1
        return redirect(url_for('index'))

    def prev(self, *args, **kwargs):
        if (self.app.config["HEAD"] == 0):
            return redirect(url_for('index'))
        self.app.config["HEAD"] = self.app.config["HEAD"] - 1
        return redirect(url_for('index'))

    def images(self, *args, **kwargs):
        f = kwargs['f']
        images = self.app.config['IMAGES']
        return send_file(images + f)

    def classify(self, *args, **kwargs):
        path = self.app.config["FILES"][self.app.config["HEAD"]]
        label = request.args.get("label")
        self.app.config["LABELS"][self.app.config["HEAD"]] = label
        
        data_name = os.path.splitext(path)[0]
        meta_path = os.path.join(self._meta_dir, f"{data_name}.json")
        label_path = os.path.join(self._label_dir, f"{data_name}.json")
        data = {}
        with open(meta_path, "r") as f:
            data = json.load(f)
        image_label = label 
        if image_label == '1':
            self.positives += 1
        else:
            self.negatives += 1
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
        return redirect(url_for('index'))

# mission = "/home/shilpag/Documents/hawk-data/dota-tennis-court-hawk_20220822-002846"        
# a = UILabeler(mission)
# a.run()
