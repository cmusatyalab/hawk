# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import io
import os
import threading
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Iterator

import numpy as np
import torch
from logzero import logger

from ...classes import (
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    ClassCounter,
    ClassLabel,
    ClassName,
    class_label_to_int,
)
from ...proto.messages_pb2 import DatasetSplit, HawkObject, LabeledTile, SendLabel
from .utils import get_example_key

if TYPE_CHECKING:
    from ...proto.messages_pb2 import DatasetSplitValue
    from ..core.mission import Mission

TMP_DIR = "test-0"
IGNORE_FILE = ["ignore", "-1", "labels"]
TRAIN_TO_TEST_RATIO = 4


class DataManager:
    def __init__(self, context: Mission):
        self._context = context
        self._staging_dir = self._context.data_dir / "examples-staging"
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        self._staging_lock = threading.Lock()
        self._examples_dir = self._context.data_dir / "examples"
        # Ignore false lint warning, DatasetSplit is EnumWrapperType instead of dict
        for example_set in list(DatasetSplit.keys()):
            example_dir = self._examples_dir / example_set.lower()
            example_dir.mkdir(parents=True, exist_ok=True)
        self._examples_lock = threading.Lock()
        self._tmp_dir = self._examples_dir / TMP_DIR
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self._example_counts: dict[str, int] = defaultdict(int)
        self._validate = self._context.check_create_test()
        logger.info(f"self . validate is {self._validate}")
        self._is_npy = False

        logger.info(f"Class list: {self._context.class_list}")
        self.class_counts = ClassCounter(self._context.class_list)

        bootstrap_zip = self._context.bootstrap_zip
        if bootstrap_zip is not None and len(bootstrap_zip):
            self.add_initial_examples(bootstrap_zip)
        self._stored_examples_event = threading.Event()
        threading.Thread(
            target=self._promote_staging_examples, name="promote-staging-examples"
        ).start()

        self._positives = 0
        self._total_positives = 0
        self._negatives = 0
        self.train_type = self._context.train_strategy
        # logger.info(f"Training strategy: {self.train_type}")
        self._radar_crop = (
            self.train_type.HasField("dnn_classifier_radar")
            and self.train_type.dnn_classifier_radar.args["pick_patches"]
        )

    def get_example_directory(self, example_set: DatasetSplitValue) -> Path:
        return self._examples_dir / self._to_dir(example_set)

    def store_labeled_tile(self, tile: LabeledTile) -> None:
        """Store the tile content along with labels in the scout"""
        if self._context.novel_class_discovery:
            ## handles saving renaming feature vectors when receiving labels and
            ## puts labels in the labels queue for future clustering.
            self.store_feature_vector(tile)
        if tile.boundingBoxes:
            self._total_positives += 1
        # logger.info(f"Original tile name: {tile.obj.objectId}")
        # logger.info(f" NEW TOTAL POSITIVES: {self._total_positives}\n\n")
        if not self._radar_crop or not tile.boundingBoxes:
            self._store_labeled_examples([tile], None)
            # logger.info("Stored negative examples...")
        else:
            with io.BytesIO(tile.obj.content) as fp:
                np_arr = np.load(fp)
            crop_list = []
            for box in tile.boundingBoxes:
                # crop each
                x, y = (
                    int(np.round(box.x * 63)),
                    int(np.round(box.y * 255)),
                    # int(np.round(box.w * 63)),
                    # int(np.round(box.h * 255)),
                )
                # predetermined crop dimensions derived from mean + 1 stdev
                # across all object instances of raddet dataset.
                crop_width, crop_height = (51, 74)
                left, right, top, bottom = (
                    max(0, x - int(np.round(crop_width / 2))),
                    min(63, x + int(np.round(crop_width / 2))),
                    max(0, y - int(np.round(crop_height / 2))),
                    min(255, y + int(np.round(crop_height / 2))),
                )
                crop_arr = np_arr[left : right + 1, top : bottom + 1, :]
                pad_left, pad_top = left, top
                pad_right, pad_bottom = 63 - right, 255 - bottom

                crop_arr_padded = np.pad(
                    crop_arr,
                    ((pad_left, pad_right), (pad_top, pad_bottom), (0, 0)),
                    mode="constant",
                )

                with io.BytesIO() as tmp:
                    np.save(tmp, crop_arr_padded)
                    crop_arr_bytes = tmp.getvalue()

                crop_tile = LabeledTile(
                    obj=HawkObject(objectId="", content=crop_arr_bytes, attributes={}),
                    boundingBoxes=[box],
                )
                crop_list.append(crop_tile)
            self._store_labeled_examples(crop_list, None)

        return

    def distribute_label(self, label: SendLabel) -> None:
        scout_index = label.scoutIndex
        if label.boundingBoxes:
            self._positives += 1
        else:
            self._negatives += 1

        if scout_index != self._context.scout_index:
            # This code should not run as not using coordinator, all labels
            # initially return to generating scout.
            logger.info(f"Fetch {label.objectId} from {scout_index}")
            stub = self._context.scouts[scout_index]
            assert stub.internal is not None
            msg = [
                b"s2s_get_tile",
                label.SerializeToString(),
            ]
            stub.internal.send_multipart(msg)
            reply = stub.internal.recv()
            if len(reply) == 0:
                obj = None
            else:
                obj = HawkObject()
                obj.ParseFromString(reply)
        else:
            # Local scout contains image of respective label received.
            obj = self._context.retriever.read_object(label.objectId)
            if self._context.novel_class_discovery and obj is not None:
                ## modify hawk obj - add fv and scout index for storage and
                ## transmission.
                obj = self.read_feature_vector(obj)

        if obj is None:
            return

        # Save labeled tile
        labeled_tile = LabeledTile(obj=obj, boundingBoxes=label.boundingBoxes)

        # save copy of original image and label to examples dir for future training
        self._context.store_labeled_tile(labeled_tile)

        # for scml, only send positives to other scouts, dont execute this if
        # using SCML staggered deployment...also need the s2s to ignore
        # negatives received if scout is currently active.  thus only idle
        # scouts receive the negative from other scouts for retraining.
        if not label.boundingBoxes:
            return

        # Transmit
        msg = [b"s2s_add_tile_and_label", labeled_tile.SerializeToString()]
        for i, stub in enumerate(self._context.scouts):  # send positives to all scouts
            if i in [self._context.scout_index, scout_index]:
                continue
            assert stub.internal is not None
            stub.internal.send_multipart(msg)
            stub.internal.recv()
        return

    def add_initial_examples(self, zip_content: bytes) -> None:
        def name_is_integer(name: str) -> bool:
            try:
                int(name)
                return True
            except ValueError:
                return False

        image_extensions = (".png", ".jpeg", ".jpg", ".npy")

        with zipfile.ZipFile(io.BytesIO(zip_content), "r") as zf:
            example_files = zf.namelist()
            for filename in example_files:
                basename = Path(filename).name
                parent_name = Path(filename).parent.name

                if basename.endswith(image_extensions) and name_is_integer(parent_name):
                    label = parent_name

                    content = zf.read(filename)
                    # logger.info(f"FILE NAME: {filename}")
                    if filename.split(".")[-1] == "npy":
                        example_file = get_example_key(content, extension=".npy")
                        self._is_npy = True
                    else:
                        example_file = get_example_key(content)
                    if self._validate and (
                        self._get_example_count(DatasetSplit.TEST, label)
                        * TRAIN_TO_TEST_RATIO
                        < self._get_example_count(DatasetSplit.TRAIN, label)
                    ):
                        example_set = DatasetSplit.TEST
                    else:
                        example_set = DatasetSplit.TRAIN

                    example_dir = os.path.join(
                        self._examples_dir,
                        DatasetSplit.Name(example_set).lower(),
                        label,
                    )

                    if not os.path.exists(example_dir):
                        os.makedirs(example_dir, exist_ok=True)

                    example_path = os.path.join(example_dir, example_file)
                    with open(example_path, "wb") as f:
                        f.write(content)

                    self._increment_example_count(example_set, label, 1)

                    # check if labels folder exists
                    label_filename = os.path.join(
                        "labels", basename.split(".")[0] + ".txt"
                    )
                    if label_filename in example_files:
                        logger.info(f"label_file {label_filename} ")
                        label_content = zf.read(label_filename)
                        label_dir = os.path.join(
                            self._examples_dir,
                            DatasetSplit.Name(example_set).lower(),
                            "labels",
                        )
                        if not os.path.exists(label_dir):
                            os.makedirs(label_dir, exist_ok=True)
                        label_path = os.path.join(
                            label_dir, example_file.split(".")[0] + ".txt"
                        )
                        with open(label_path, "wb") as f:
                            f.write(label_content)

                        # Do we want to parse label_content and properly count all
                        # bounding boxes?
                        self.class_counts.update({POSITIVE_CLASS: 1})
                    else:
                        # add single sample to respective class
                        class_label = ClassLabel(int(label))
                        self.class_counts.count(class_label)

        logger.info(
            f"New positives {self.class_counts.positives}, "
            f"negatives {self.class_counts.negatives}, "
            f"by class: {self.class_counts!r}"
        )

        # Skip training if we already have a bootstrap model
        retrain = not self._context.check_initial_model()
        logger.info(
            f"Initial model {self._context.check_initial_model()} retrain {retrain}"
        )
        self._context.new_labels_callback(self.class_counts, retrain=retrain)

    @contextmanager
    def get_examples(self, example_set: DatasetSplitValue) -> Iterator[Path]:
        assert example_set is not DatasetSplit.TEST
        with self._examples_lock:
            example_dir = self._examples_dir / self._to_dir(example_set)
            yield example_dir

    def reset(self, train_only: bool) -> None:
        with self._staging_lock:
            self._clear_dir(self._staging_dir, train_only)

        with self._examples_lock:
            self._clear_dir(self._examples_dir, train_only)

    def _clear_dir(self, dir_path: Path, train_only: bool) -> None:
        for child in dir_path.iterdir():
            if child.is_dir():
                if child.name != "test" or not train_only:
                    self._clear_dir(child, train_only)
            else:
                child.unlink()

    def _class_to_label(self, class_name: ClassName) -> ClassLabel:
        self._context.class_list.add(class_name)
        return self._context.class_list.index(class_name)

    def _store_labeled_examples(
        self,
        examples: Iterable[LabeledTile],
        callback: Callable[[LabeledTile], None] | None,
    ) -> None:
        with self._staging_lock:
            # logger.info(
            #    "Grabbed staging lock in store labeled examples... "
            #    f"for {len(examples)} examples, {time.time()}"
            # )
            old_dirs = []
            for dir in self._staging_dir.iterdir():
                if dir.name not in IGNORE_FILE:
                    for lbl in dir.iterdir():
                        old_dirs.append(lbl)

            for example in examples:
                obj = example.obj
                if self._is_npy:
                    example_file = get_example_key(obj.content, extension=".npy")
                else:
                    example_file = get_example_key(obj.content)
                self._remove_old_paths(example_file, old_dirs)

                if not example.boundingBoxes:
                    # negative sample
                    label: ClassLabel | None = ClassLabel(0)
                    counts = {NEGATIVE_CLASS: 1}
                elif (
                    example.boundingBoxes[0].w == 1.0
                    and example.boundingBoxes[0].h == 1.0
                ):
                    # classification
                    class_name = ClassName(example.boundingBoxes[0].class_name)
                    label = self._class_to_label(class_name)
                    counts = {class_name: 1}
                else:
                    # detection
                    label = ClassLabel(1)
                    counts = {POSITIVE_CLASS: 1}
                    # Alternatively if we want to count individual detections...
                    # counts = Counter(
                    #     ClassName(bbox.class_name) for bbox in example.boundingBoxes
                    # )

                self.class_counts.update(counts)

                if self._validate:
                    example_subdir = self._staging_dir / "unspecified"
                else:
                    example_subdir = self._staging_dir / self._to_dir(
                        DatasetSplit.TRAIN
                    )

                if label is not None:
                    # 0 or 1 or ...
                    example_path = example_subdir / str(label) / example_file
                    example_path.parent.mkdir(parents=True, exist_ok=True)
                    if self._radar_crop:
                        with io.BytesIO(obj.content) as fp:
                            arr = np.load(fp)
                        arr = arr.reshape((64, 256, 3))
                        np.save(example_path, arr)
                    else:
                        example_path.write_bytes(obj.content)

                    label_path = (example_subdir / "labels" / example_file).with_suffix(
                        ".txt"
                    )
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    with label_path.open("w") as f:
                        for bbox in example.boundingBoxes:
                            class_name = ClassName(bbox.class_name)
                            class_label = self._class_to_label(class_name)

                            # -1 because yolo counts positive classes starting from 0
                            index = class_label_to_int(class_label) - 1
                            f.write(f"{index} {bbox.x} {bbox.y} {bbox.w} {bbox.h}\n")
                else:
                    ignore_file = self._staging_dir / IGNORE_FILE[0]
                    with ignore_file.open("a+") as f:
                        f.write(example_file + "\n")

                if callback is not None:
                    callback(example)
        self._stored_examples_event.set()

    def _promote_staging_examples(self) -> None:
        while not self._context._abort_event.is_set():
            try:
                self._stored_examples_event.wait()
                self._stored_examples_event.clear()

                new_samples = ClassCounter(self._context.class_list)

                with self._examples_lock:
                    set_dirs = {}
                    for example_set in [DatasetSplit.TRAIN, DatasetSplit.TEST]:
                        example_dir = self._examples_dir / self._to_dir(example_set)
                        set_dirs[example_set] = list(example_dir.iterdir())
                    with self._staging_lock:
                        for file in self._staging_dir.iterdir():
                            if file.name == IGNORE_FILE[0]:
                                with file.open() as ignore_file:
                                    for line in ignore_file:
                                        for example_set in set_dirs:
                                            old_path = self._remove_old_paths(
                                                line, set_dirs[example_set]
                                            )
                                            if old_path is not None:
                                                self._increment_example_count(
                                                    example_set,
                                                    old_path.parent.name,
                                                    -1,
                                                )
                            elif (
                                file.name not in IGNORE_FILE
                            ):  # to exclude easy-negative directory
                                self._promote_staging_examples_dir(
                                    file, set_dirs, new_samples
                                )

                logger.info(
                    f"Promoted staging examples, totals by class={self.class_counts!r}"
                )
                if not self._context._abort_event.is_set() and new_samples:
                    self._context.new_labels_callback(new_samples)

            except Exception as e:
                logger.exception(e)

    def _promote_staging_examples_dir(
        self,
        subdir: Path,
        set_dirs: dict[DatasetSplitValue, list[Path]],
        new_samples: ClassCounter,
    ) -> None:
        assert (
            subdir.name == self._to_dir(DatasetSplit.TRAIN)
            or subdir.name == self._to_dir(DatasetSplit.TEST)
            or subdir.name == "unspecified"
        )

        ### create simple list of length num classes, which is reset to zero each time
        for label in subdir.iterdir():  ## label is 0 1 ... labels
            # labels will get moved along with their data
            if label.name == "labels":
                continue

            example_files = list(label.iterdir())

            class_label = ClassLabel(int(label.name))
            new_samples.count(class_label, len(example_files))

            for example_file in example_files:
                for example_set in set_dirs:
                    old_path = self._remove_old_paths(
                        example_file.name, set_dirs[example_set]
                    )
                    if old_path is not None:
                        self._increment_example_count(
                            example_set, old_path.parent.name, -1
                        )

                if subdir.name == "test" or (
                    subdir.name == "unspecified"
                    and self._get_example_count(DatasetSplit.TEST, label.name)
                    * TRAIN_TO_TEST_RATIO
                    < self._get_example_count(DatasetSplit.TRAIN, label.name)
                ):
                    example_set = DatasetSplit.TEST
                else:
                    example_set = DatasetSplit.TRAIN

                self._increment_example_count(example_set, label.name, 1)

                example_set_path = self._examples_dir / self._to_dir(example_set)
                example_path = example_set_path / label.name / example_file.name
                example_path.parent.mkdir(parents=True, exist_ok=True)
                example_file.rename(example_path)

                # move associated labels/<file_stem>.txt file
                label_file = (subdir / "labels" / example_file.name).with_suffix(".txt")
                if label_file.exists():
                    label_path = example_set_path / "labels" / label_file.name
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    label_file.rename(label_path)

    def _get_example_count(self, example_set: DatasetSplitValue, label: str) -> int:
        return self._example_counts[f"{DatasetSplit.Name(example_set)}_{label}"]

    def _increment_example_count(
        self, example_set: DatasetSplitValue, label: str, delta: int
    ) -> None:
        self._example_counts[f"{DatasetSplit.Name(example_set)}_{label}"] += delta

    @staticmethod
    def _remove_old_paths(example_file: str, old_dirs: list[Path]) -> Path | None:
        for old_path in old_dirs:
            old_example_path = old_path / example_file
            if old_example_path.exists():
                old_example_path.unlink()
                return old_example_path

        return None

    @staticmethod
    def _to_dir(example_set: DatasetSplitValue) -> str:
        return DatasetSplit.Name(example_set).lower()

    def store_feature_vector(self, tile: LabeledTile) -> None:
        obj = tile.obj
        label = tile.boundingBoxes
        base = f"{Path(obj.objectId).stem}.pt"
        if not label:
            label_dir = "0"
        else:
            class_name = ClassName(label[0].class_name)
            label_dir = str(self._class_to_label(class_name))  ## get the label
        label_file_path = self._context._feature_vector_dir / label_dir / base
        label_file_path.parent.mkdir(parents=True, exist_ok=True)

        if (
            int(obj.attributes["source_scout_index"].decode())
            == self._context.scout_index
        ):  ## receiving a label from this scout
            temp_file_path = self._context._feature_vector_dir / "temp" / base
            assert os.path.exists(temp_file_path)
            ## move fv from temp to label dir once label received.
            os.rename(temp_file_path, label_file_path)
        else:
            if obj.attributes["feature_vector"] is not None:
                vector: torch.Tensor = torch.load(
                    io.BytesIO(obj.attributes["feature_vector"])
                )
                torch.save(vector, label_file_path)
                ## saves the feature vector of any labeled sample received by
                ## another scout.

        ## send only relevant info to novel class labels queue for future
        ## clustering.
        label_tuple = (label_dir, label_file_path)
        self._context.labels_queue.put(label_tuple)

    def read_feature_vector(self, obj: HawkObject) -> HawkObject:
        ## read the feature vector from its temp location and add to HawkObject
        ## for transmission to other scouts.
        temp_dir = self._context._feature_vector_dir / "temp"
        feature_vector_path = (temp_dir / Path(obj.objectId).stem).with_suffix(".pt")
        feature_vector = torch.load(feature_vector_path)
        with io.BytesIO() as fv_bytes:
            torch.save(feature_vector, fv_bytes)
            feat_vect = fv_bytes.getvalue()

        obj.attributes["feature_vector"] = feat_vect
        obj.attributes["source_scout_index"] = str.encode(
            str(self._context.scout_index)
        )
        return obj
