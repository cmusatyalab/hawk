# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import io
import os
import threading
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Iterable, Callable, Optional, Tuple

from logzero import logger

from hawk.context.data_manager_context import DataManagerContext
from hawk.proto.messages_pb2 import DatasetSplit, LabelWrapper, LabeledTile, HawkObject
from hawk.core.utils import get_example_key

TMP_DIR = 'test-0'

IGNORE_FILE = ['ignore', '-1', 'labels']


class DataManager(object):

    def __init__(self, context: DataManagerContext):
        self._context = context
        self._staging_dir = self._context.data_dir / 'examples-staging'
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        self._staging_lock = threading.Lock()
        self._examples_dir = self._context.data_dir / 'examples'
        for example_set in DatasetSplit.keys():
            example_dir = self._examples_dir / example_set.lower()
            example_dir.mkdir(parents=True, exist_ok=True)
        self._examples_lock = threading.Lock()
        self._tmp_dir = self._examples_dir / TMP_DIR
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self._example_counts = defaultdict(int)
        bootstrap_zip = self._context.bootstrap_zip
        if bootstrap_zip is not None and len(bootstrap_zip):
            self.add_initial_examples(bootstrap_zip)
        self._stored_examples_event = threading.Event()
        threading.Thread(target=self._promote_staging_examples,
                         name='promote-staging-examples').start()
        self._positives = 0  
        self._negatives = 0 

    def get_example_directory(self, example_set: DatasetSplit) -> Path:
        return self._examples_dir / self._to_dir(example_set)

    def store_labeled_tile(self, tile: LabeledTile) -> None:
        """Store the tile content along with labels in the scout
        """
        self._store_labeled_examples([tile], None)
        return

    def distribute_label(self, label: LabelWrapper) -> None:
        scout_index = label.scoutIndex
        if label.imageLabel == "0":
            self._negatives += 1
        else:
            self._positives += 1
        if  int(scout_index) != int(self._context.scout_index):
            logger.info("Fetch {} from {}".format(label.objectId, scout_index))
            stub = self._context.scouts[scout_index]
            msg = {
                "method": "s2s_get_tile",
                "msg": label.SerializeToString(),
            }
            stub.internal.send_pyobj(msg)
            reply = stub.internal.recv()
            if not len(reply):
                object = None
            else:
                object = HawkObject()
                object.ParseFromString(reply)
        else:
            object = self._context.retriever.get_object(
                object_id=label.objectId 
            )
        
        if object is None:
            return
        # Save labeled tile 
        labeled_tile = LabeledTile(
            object=object,
            label=label,
            )
        self._context.store_labeled_tile(labeled_tile)
        if label.imageLabel == "0":
            return
        # Transmit 
        for i, stub in enumerate(self._context.scouts):
            if i in [self._context.scout_index, scout_index]:
                continue 
            msg = {
                "method": "s2s_add_tile_and_label",
                "msg": labeled_tile.SerializeToString(),
            }
            stub.internal.send_pyobj(msg)
            stub.internal.recv()
        
        return 
        
    def add_initial_examples(self, zip_content):
        def name_is_integer(name: str):
            try:
                int(name)
                return True
            except ValueError:
                return False

        image_extensions = ('.png', '.jpeg', '.jpg')
        labels = []
        with zipfile.ZipFile(io.BytesIO(zip_content), "r") as zf:
            example_files = zf.namelist()
            for filename in example_files:
                basename = Path(filename).name
                parent_name = Path(filename).parent.name
                    
                if basename.endswith(image_extensions) and name_is_integer(parent_name):
                    label = parent_name
                    content = zf.read(filename)
                    example_file = get_example_key(content)
                    train_dir = os.path.join(self._examples_dir, 'train', str(label))
                    if not os.path.exists(train_dir):
                        os.makedirs(train_dir, exist_ok=True)
                    example_path = os.path.join(train_dir, example_file)
                    with open(example_path, 'wb') as f:
                        f.write(content)

                    # check if labels folder exists
                    label_filename = os.path.join('labels', basename.split('.')[0]+".txt")
                    if label_filename in example_files:
                        logger.info("label_file {} ".format(label_filename))
                        label_content = zf.read(label_filename)
                        train_dir = os.path.join(self._examples_dir, 'train', 'labels')
                        if not os.path.exists(train_dir):
                            os.makedirs(train_dir, exist_ok=True)
                        example_path = os.path.join(train_dir, example_file.split('.')[0]+".txt")
                        with open(example_path, 'wb') as f:
                            f.write(label_content)
                        
                    labels.append(int(label))

        new_positives = sum(labels)
        new_negatives = len(labels) - new_positives
        print(" New positives {} \n Negatives {}".format(new_positives, new_negatives))

        retrain = True
        if self._context.check_initial_model():
            retrain = False
        logger.info("Initial model {} retrain {}".format(self._context.check_initial_model(), retrain))
        self._context.new_examples_callback(new_positives, new_negatives, retrain=retrain)


    @contextmanager
    def get_examples(self, example_set: DatasetSplit) -> Iterable[Path]:
        with self._examples_lock:
            if self._context.scout_index != 0:
                yield self._examples_dir / self._to_dir(example_set)
            else:
                example_dir = self._examples_dir / self._to_dir(example_set)
                if example_set is DatasetSplit.TEST:
                    for label in example_dir.iterdir():
                        tmp_label_dir = self._tmp_dir / label.name
                        tmp_label_dir.mkdir(parents=True, exist_ok=True)
                        test_files = list(label.iterdir())
                        for i in range(0, len(test_files), len(self._context.scouts)):
                            test_files[i].rename(tmp_label_dir / test_files[i].name)

                    yield self._tmp_dir

                    for label in self._tmp_dir.iterdir():
                        for tmp_file in label.iterdir():
                            tmp_file.rename(example_dir / label.name / tmp_file.name)
                else:
                    yield example_dir


    def get_example_path(self, example_set: DatasetSplit, label: str, example: str) -> Path:
        # assert self._examples_lock.locked()
        assert self._context.scout_index == 0
        locked = self._examples_lock.locked()
        if locked :
            return self._examples_dir / self._to_dir(example_set) / label / example
        with self._examples_lock:
            return self._examples_dir / self._to_dir(example_set) / label / example

    def reset(self, train_only: bool):
        with self._staging_lock:
            self._clear_dir(self._staging_dir, train_only)

        with self._examples_lock:
            self._clear_dir(self._examples_dir, train_only)

    def _clear_dir(self, dir_path: Path, train_only: bool):
        for child in dir_path.iterdir():
            if child.is_dir():
                if child.name != 'test' or not train_only:
                    self._clear_dir(child, train_only)
            else:
                child.unlink()

    def _store_labeled_examples(self, examples: Iterable[LabeledTile],
                                callback: Optional[Callable[[LabeledTile], None]]) -> None:
        with self._staging_lock:
            old_dirs = []
            for dir in self._staging_dir.iterdir():
                if dir.name not in IGNORE_FILE:
                    for label in dir.iterdir():
                        old_dirs.append(label)

            for example in examples:
                object = example.object
                example_file = get_example_key(object.content)
                self._remove_old_paths(example_file, old_dirs)

                label =  example.label.imageLabel
                bounding_boxes = example.label.boundingBoxes
                split = DatasetSplit.TRAIN
                if label != '-1':
                    example_subdir = self._to_dir(split)

                    label_dir = self._staging_dir / example_subdir / label
                    label_dir.mkdir(parents=True, exist_ok=True)
                    example_path = label_dir / example_file
                    with example_path.open('wb') as f:
                        f.write(object.content)
                    if bounding_boxes:
                        label_dir = self._staging_dir / example_subdir / 'labels'
                        label_dir.mkdir(parents=True, exist_ok=True)
                        example_path = label_dir / (example_file.split('.')[0] + ".txt")                        
                        with example_path.open('w') as f:
                            f.write("\n".join(bounding_boxes))
                        
                else:
                    logger.info('Example set to ignore - skipping')
                    ignore_file = self._staging_dir / IGNORE_FILE[0]
                    with ignore_file.open('a+') as f:
                        f.write(example_file + '\n')

                if callback is not None:
                    callback(example)

        self._stored_examples_event.set()


    def _promote_staging_examples(self):
        while not self._context._abort_event.is_set():
            try:
                self._stored_examples_event.wait()
                self._stored_examples_event.clear()

                new_positives = 0
                new_negatives = 0
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
                                            old_path = self._remove_old_paths(line, set_dirs[example_set])
                                            if old_path is not None:
                                                self._increment_example_count(example_set, old_path.parent.name, -1)
                            elif file.name not in IGNORE_FILE:  # to exclude easy-negative directory
                                dir_positives, dir_negatives = self._promote_staging_examples_dir(file, set_dirs)
                                new_positives += dir_positives
                                new_negatives += dir_negatives

                if not self._context._abort_event.is_set():
                    self._context.new_examples_callback(new_positives, new_negatives)
            except Exception as e:
                logger.exception(e)

    def _promote_staging_examples_dir(self, subdir: Path, set_dirs: Dict['DatasetSplit', List[Path]]) -> Tuple[int, int]:
        assert subdir.name == self._to_dir(DatasetSplit.TRAIN) \
               or subdir.name == self._to_dir(DatasetSplit.TEST) \
               or subdir.name == 'unspecified'

        new_positives = 0
        new_negatives = 0

        if subdir.name == 'test':
            return new_positives, new_negatives

        for label in subdir.iterdir():
            example_files = list(label.iterdir())
            if subdir.name != 'test':
                if label.name == '1':
                    new_positives += len(example_files)
                elif label.name == 'labels':
                    pass
                else:
                    new_negatives += len(example_files)

            for example_file in example_files:
                for example_set in set_dirs:
                    old_path = self._remove_old_paths(example_file.name, set_dirs[example_set])
                    if old_path is not None:
                        self._increment_example_count(example_set, old_path.parent.name, -1)

                example_set = DatasetSplit.TRAIN

                self._increment_example_count(example_set, label.name, 1)
                example_dir = self._examples_dir / self._to_dir(example_set) / label.name
                example_dir.mkdir(parents=True, exist_ok=True)
                example_path = example_dir / example_file.name
                example_file.rename(example_path)

        return new_positives, new_negatives

    def _get_example_count(self, example_set: DatasetSplit, label: str) -> int:
        return self._example_counts['{}_{}'.format(DatasetSplit.Name(example_set), label)]

    def _increment_example_count(self, example_set: DatasetSplit, label: str, delta: int) -> None:
        self._example_counts['{}_{}'.format(DatasetSplit.Name(example_set), label)] += delta

    @staticmethod
    def _remove_old_paths(example_file: str, old_dirs: List[Path]) -> Optional[Path]:
        for old_path in old_dirs:
            old_example_path = old_path / example_file
            if old_example_path.exists():
                old_example_path.unlink()
                logger.info('Removed old path {} for example'.format(old_example_path))
                return old_example_path

        return None

    @staticmethod
    def _to_dir(example_set: DatasetSplit):
        return DatasetSplit.Name(example_set).lower()