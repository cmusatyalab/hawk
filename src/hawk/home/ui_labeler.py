# SPDX-FileCopyrightText: 2022-2023 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Any, Callable

import numpy as np
from flask import (
    Flask,
    Response,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from flask.typing import ResponseReturnValue
from logzero import logger
from werkzeug import Response as WerkzeugResponse

from .typing import Labeler, LabelQueueType, MetaQueueType, StatsQueueType


@dataclass
class ImageBox:
    id: int
    xMin: int
    xMax: int
    yMin: int
    yMax: int

    # convert to yolo fomat
    def yolo(self, class_id: int = 0) -> str:
        return f"{class_id} {self.xMin} {self.xMax} {self.yMin} {self.yMax}"


class EndpointAction:
    def __init__(self, action: Callable[..., ResponseReturnValue]):
        self.action = action
        self.response = Response(status=200, headers={})

    def __call__(self, *args: Any, **kwargs: Any) -> ResponseReturnValue:
        # Perform the action
        response = self.action(*args, **kwargs)
        if response is not None:
            return response
        else:
            return self.response


class UILabeler(Labeler):
    def __init__(self, mission_dir: Path, save_automatically: bool = False):
        self.app = Flask(__name__)
        self.app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

        self._image_dir = mission_dir / "images"
        self._meta_dir = mission_dir / "meta"
        self._label_dir = mission_dir / "labels"
        self._stats_dir = mission_dir / "logs"
        self._label_map = {
            "-1": {"color": "#ffffff", "text": "UNLABELED"},
            "0": {"color": "#ae163e", "text": "NEGATIVE"},
            "1": {"color": "green", "text": "POSITIVE"},
        }
        self._stats_keys = {
            "version": "Model Version : ",
            "totalObjects": "Total Tiles : ",
            "processedObjects": "Tiles Processed : ",
            "positives": "Positives Labeled : ",
            "negatives": "Negatives Labeled : ",
        }

        self.app.config["IMAGES"] = self._image_dir
        self.app.config["LABELS"] = []
        self.app.config["HEAD"] = 0
        self.app.config["LOGS"] = self._stats_dir
        self.label_changed = False
        self.save_auto = save_automatically

        files = sorted(file for file in self._image_dir.iterdir() if file.is_file())
        if not files:
            logger.error("No files")
            exit()

        self.app.config["LABELS"] = ["-1"] * len(files)
        self.app.config["FILES"] = files
        self.not_end = True

        self.image_boxes: list[ImageBox] = []
        self.num_thumbnails = 4
        self.add_all_endpoints()

    def start_labeling(
        self,
        input_q: MetaQueueType,
        result_q: LabelQueueType,
        stats_q: StatsQueueType,
        stop_event: Event,
    ) -> None:
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

    def run(self) -> None:
        self.result_q = mp.Queue()
        self.stats_q = mp.Queue()
        self.positives = 0
        self.negatives = 0
        self.bytes = 0
        self.app.jinja_env.filters["bool"] = bool
        self.app.run(port=8000)

    def add_all_endpoints(self) -> None:
        # Add endpoints
        self.add_endpoint(endpoint="/", endpoint_name="index", handler=self.index)
        self.add_endpoint(endpoint="/next", endpoint_name="next", handler=self.next)
        self.add_endpoint(endpoint="/prev", endpoint_name="prev", handler=self.prev)
        self.add_endpoint(
            endpoint="/backward", endpoint_name="backward", handler=self.backward
        )
        self.add_endpoint(
            endpoint="/forward", endpoint_name="forward", handler=self.forward
        )
        self.add_endpoint(endpoint="/save", endpoint_name="save", handler=self.save)
        self.add_endpoint(endpoint="/undo", endpoint_name="undo", handler=self.undo)
        self.add_endpoint(endpoint="/add/<id>", endpoint_name="add", handler=self.add)
        self.add_endpoint(
            endpoint="/remove/<id>", endpoint_name="remove", handler=self.remove
        )
        self.add_endpoint(
            endpoint="/image/<f>", endpoint_name="images", handler=self.images
        )
        self.add_endpoint(
            endpoint="/classify/<id>", endpoint_name="classify", handler=self.classify
        )

    def add_endpoint(
        self,
        endpoint: str,
        endpoint_name: str,
        handler: Callable[..., ResponseReturnValue],
    ) -> None:
        self.app.add_url_rule(endpoint, endpoint_name, EndpointAction(handler))
        # You can also add options here : "... , methods=['POST'], ... "

    # ==================== ------ API Calls ------- ====================
    def index(self) -> str:
        self.reload_directory()
        if len(self.app.config["FILES"]) == 0:
            self.not_end = False
            self.app.config["HEAD"] = -1
            return render_template(
                "index.html",
                directory="",
                images=[""],
                head=0,
                len=len(self.app.config["FILES"]),
                color="#ffffff",
                label_text="",
                boxes=[asdict(box) for box in self.image_boxes],
                stats={},
                label_changed=1 if self.label_changed else 0,
            )

        index_num = self.app.config["HEAD"]
        length_files = len(self.app.config["FILES"])

        # main image and thumbnails
        image_paths = []
        for i in range(self.num_thumbnails + 1):
            idx = index_num + i
            if idx >= length_files:
                break
            image_paths.append(self.app.config["FILES"][idx])

        if self.app.config["HEAD"] == 0 and not self.label_changed:
            label, boxes = self.read_labels()
            self.app.config["LABELS"][self.app.config["HEAD"]] = label
            self.image_boxes = boxes
        else:
            label = self.app.config["LABELS"][self.app.config["HEAD"]]

        color = self._label_map[label]["color"]
        label_text = self._label_map[label]["text"]
        self.not_end = not (self.app.config["HEAD"] == length_files - 1)
        condition_changed = self.label_changed and not self.save_auto
        search_stats = self.get_latest_stats()

        return render_template(
            "index.html",
            images=image_paths,
            head=self.app.config["HEAD"] + 1,
            files=len(self.app.config["FILES"]),
            color=color,
            label_text=label_text,
            boxes=self.image_boxes,
            stats=search_stats,
            label_changed=int(condition_changed is True),
            save=int(self.save_auto is True),
        )

    def reload_directory(self) -> None:
        old_length = len(self.app.config["FILES"])
        files = sorted(
            file for file in self.app.config["IMAGES"].walk() if file.is_file()
        )
        if not files:
            logger.error("No files")
            exit()
        new_length = len(files)
        new_files = new_length - old_length
        self.app.config["LABELS"].expand(["-1"] * new_files)
        self.app.config["FILES"] = files
        self.not_end = not (self.app.config["HEAD"] == new_length - 1)
        if new_files and self.app.config["HEAD"] < 0:
            self.app.config["HEAD"] = 0

    def read_labels(self, head_id: int = -1) -> tuple[str, list[ImageBox]]:
        if head_id == -1:
            head_id = self.app.config["HEAD"]

        path = self.app.config["FILES"][head_id]
        label_path = self._label_dir / f"{path.stem}.json"

        if label_path.exists():
            with open(label_path) as f:
                image_labels = json.load(f)

            label = str(image_labels["imageLabel"])
            assert label in ["1", "0"], f"Unknown label {label}"
            image_boxes = image_labels["boundingBoxes"]
            boxes = []
            for i, box in enumerate(image_boxes, 1):
                _, xMin, xMax, yMin, yMax = box.split(" ")
                boxes.append(ImageBox(id=i, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax))

            return label, boxes

        return "-1", []

    def write_labels(self) -> None:
        if not self.label_changed:
            return
        path = self.app.config["FILES"][self.app.config["HEAD"]]
        meta_path = self._meta_dir / f"{path.stem}.json"
        label_path = self._label_dir / f"{path.stem}.json"
        image_label = self.app.config["LABELS"][self.app.config["HEAD"]]

        if image_label == "-1":
            label_path.unlink(missing_ok=True)

        if image_label not in ["1", "0"]:
            return

        data = {}
        with open(meta_path) as f:
            data = json.load(f)
        self.bytes += data["size"]

        # assuming binary class hardcoding positive id as 0
        bounding_boxes = [box.yolo(class_id=0) for box in self.image_boxes]
        label = {
            "objectId": data["objectId"],
            "scoutIndex": data["scoutIndex"],
            "imageLabel": image_label,
            "boundingBoxes": bounding_boxes,
        }
        with open(label_path, "w") as f:
            json.dump(label, f, indent=4, sort_keys=True)

        self.result_q.put(str(label_path))
        image_labels = np.array(self.app.config["LABELS"])
        self.positives = len(np.where(image_labels == "1")[0])
        self.negatives = len(np.where(image_labels == "0")[0])
        logger.info(
            "({}, {}) Labeled {}".format(
                self.positives, self.negatives, data["objectId"]
            )
        )
        self.stats_q.put((self.positives, self.negatives, self.bytes))
        self.label_changed = False

    def save(self) -> WerkzeugResponse:
        self.write_labels()
        return redirect(url_for("index"))

    def undo(self) -> WerkzeugResponse:
        self.label_changed = True
        label = "-1"
        self.app.config["LABELS"][self.app.config["HEAD"]] = label
        if label != "1":
            self.image_boxes = []
        self.write_labels()
        self.label_changed = False
        return redirect(url_for("index"))

    def next(self) -> WerkzeugResponse:
        if self.save_auto:
            self.write_labels()
        if self.not_end:
            self.app.config["HEAD"] = self.app.config["HEAD"] + 1
        else:
            logger.info("Waiting for Results ...")
            while not self.not_end:
                time.sleep(1)
                self.reload_directory()

        label, boxes = self.read_labels()
        self.app.config["LABELS"][self.app.config["HEAD"]] = label
        self.image_boxes = boxes
        self.label_changed = False
        return redirect(url_for("index"))

    def backward(self) -> WerkzeugResponse:
        if self.save_auto:
            self.write_labels()

        found_positive = False
        curr_id = self.app.config["HEAD"]
        new_id = curr_id
        # length_files = len(self.app.config["FILES"])

        while not found_positive and new_id - 1 >= 0:
            new_id -= 1
            label, boxes = self.read_labels(new_id)
            if label == "1":
                found_positive = True

        if found_positive:
            self.app.config["HEAD"] = new_id

        label, boxes = self.read_labels()
        self.app.config["LABELS"][self.app.config["HEAD"]] = label
        self.image_boxes = boxes
        self.label_changed = False

        return redirect(url_for("index"))

    def forward(self) -> WerkzeugResponse:
        if self.save_auto:
            self.write_labels()

        found_positive = False
        curr_id = self.app.config["HEAD"]
        new_id = curr_id
        length_files = len(self.app.config["FILES"])

        while not found_positive and new_id + 1 < length_files:
            new_id += 1
            label, boxes = self.read_labels(new_id)
            if label == "1":
                found_positive = True

        if found_positive:
            self.app.config["HEAD"] = new_id

        label, boxes = self.read_labels()
        self.app.config["LABELS"][self.app.config["HEAD"]] = label
        self.image_boxes = boxes
        self.label_changed = False

        return redirect(url_for("index"))

    def prev(self) -> WerkzeugResponse:
        if self.app.config["HEAD"] == 0:
            return redirect(url_for("index"))
        self.app.config["HEAD"] = self.app.config["HEAD"] - 1
        label, boxes = self.read_labels()
        self.app.config["LABELS"][self.app.config["HEAD"]] = label
        self.image_boxes = boxes
        self.label_changed = False
        return redirect(url_for("index"))

    def add(self, **kwargs: Any) -> WerkzeugResponse:
        id = int(kwargs["id"])
        self.label_changed = True
        self.app.config["LABELS"][self.app.config["HEAD"]] = "1"
        xMin = int(request.args.get("xMin", 0))
        xMax = int(request.args.get("xMax", 0))
        yMin = int(request.args.get("yMin", 0))
        yMax = int(request.args.get("yMax", 0))
        self.image_boxes.append(
            ImageBox(id=id, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax)
        )
        return redirect(url_for("index"))

    def remove(self, **kwargs: Any) -> WerkzeugResponse:
        id = int(kwargs["id"])
        self.label_changed = True
        index = id - 1
        del self.image_boxes[index]
        for label in self.image_boxes[index:]:
            label.id = label.id - 1
        if not len(self.image_boxes):
            self.app.config["LABELS"][self.app.config["HEAD"]] = "-1"
        return redirect(url_for("index"))

    def images(self, **kwargs: Any) -> WerkzeugResponse:
        f = kwargs["f"]
        return send_from_directory(self.app.config["IMAGES"], f)

    def classify(self) -> WerkzeugResponse:
        self.label_changed = True
        label = request.args.get("label")
        self.app.config["LABELS"][self.app.config["HEAD"]] = label
        if label != "1":
            self.image_boxes = []
        if label == "-1":
            self.undo()

        return redirect(url_for("index"))

    def get_latest_stats(self) -> list[str]:
        list_of_files = self._stats_dir.glob("stats-*")
        latest_stats = max(list_of_files, key=os.path.getctime)
        search_stats = {}
        with open(latest_stats) as f:
            search_stats = json.load(f)
        stats = []
        for k, v in self._stats_keys.items():
            stats_item = search_stats.get(k, 0)
            if k == "positives":
                stats_item = max(stats_item, self.positives)
            elif k == "negatives":
                stats_item = max(stats_item, self.negatives)
            stats_text = v + str(int(stats_item))
            stats.append(stats_text)
        return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mission")
    args = parser.parse_args()
    a = UILabeler(args.mission)
    a.run()

# python -m hawk.home.ui_labeler \
#     /home/shilpag/Documents/label_ui/dota-20-30k_swimming-pool_30_20221031-121302
# python -m hawk.home.ui_labeler \
#     School_Bus_Hawk/Hawk_Mission_Data/test-hawk-school_bus_tiles_video_20221010-131323
