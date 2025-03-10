# SPDX-FileCopyrightText: 2022,2024 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from fabric import Connection

from ..mission_config import MissionConfig, load_config

# from invoke.exceptions import UnexpectedExit


def shuffle_and_generate_index_files(
    config: MissionConfig,
    split_by_prefix: bool = False,
    random_seed: int | None = None,
    dry_run: bool = False,
) -> int:
    """
    Read stream.txt from input dir
    filter out (accidental) duplicates
    reshuffle
    split into per-scout, non-overlapping, index files
    send index files to the scouts
    """
    data_config = config["dataset"]
    stream_file = Path(data_config["stream_path"])
    index_file = data_config["index_path"]
    hidden_class = data_config.get("hidden_class", False)
    hidden_class_path = data_config.get("hidden_class_path", "")
    hidden_class_start = data_config.get("hidden_class_start", 0)
    # hidden_class_name = data_config.get("hidden_class_name", 2)
    hosts = config.scouts
    assert stream_file.exists(), "Stream file does not exist"

    print("** Loading unique entries from stream file")

    contents = sorted({line.rstrip() for line in stream_file.open()})
    total_tiles = len(contents)

    print(f"** Shuffling {total_tiles} items")

    random.seed(random_seed)
    random.shuffle(contents)

    print("** Counting frames")

    img_tile_map = defaultdict(list)
    # img_label_map = defaultdict(list)

    if split_by_prefix:
        """Group into frames by common prefix"""
        unique_keys = set()
        for content in contents:
            path, label = content.rsplit(maxsplit=1)
            k = Path(path).name.split("_")[0]
            unique_keys.add(k)
        keys = list(unique_keys)
    else:
        """Randomly group into frames based on tiles_per_frame config option"""
        num_tiles_per_frame = data_config.get("tiles_per_frame", 200)
        num_frames = math.ceil(total_tiles / num_tiles_per_frame)
        keys = [str(frame) for frame in range(num_frames)]

    total_keys = len(keys)
    items_per_frame = total_tiles // total_keys

    print(
        f"** Splitting items over {total_keys} frames (~{items_per_frame} items/frame)"
    )

    # tiles_per_frame = math.ceil(total_tiles / total_keys)
    per_frame = np.array_split(contents, total_keys)

    for i, tiles_per_frame in enumerate(per_frame):
        k = keys[i]
        for content in tiles_per_frame:
            path, label = content.split()
            img_tile_map[k].append(content)
            # img_label_map[k].append(int(label))

    keys = list(img_tile_map.keys())
    random.shuffle(keys)

    num_hosts = len(hosts)
    frames_per_host = total_keys // num_hosts
    min_items_per_host = frames_per_host * items_per_frame
    max_items_per_host = (frames_per_host + 1) * (items_per_frame + 1)

    print(
        f"** Splitting frames across {num_hosts} hosts (~{frames_per_host} frames/host,"
        f" ~{min_items_per_host}-{max_items_per_host} items/host)"
    )

    div_keys = [keys[i::num_hosts] for i in range(num_hosts)]
    div_files = []

    for keys in div_keys:
        paths = []
        for k in keys:
            files = img_tile_map[k]
            for f in files:
                paths.append(f)

        div_files.append(paths)

    # Insert hidden samples here according to the number and desired start
    # position (25%, 50%, etc.)
    if hidden_class:
        ## load the samples from the file with containing hidden samples
        hidden_samples = []
        with open(hidden_class_path) as fp:
            for sample in fp.read().splitlines():
                hidden_samples.append(sample)

        hidden_sample_set = hidden_samples[
            : int(len(hidden_samples) * (1 - float(hidden_class_start)))
        ]
        for hidden_sample in hidden_sample_set:

            ## pick random scout and random index

            scout = random.randint(0, len(hosts) - 1)
            index = random.randint(
                int(float(hidden_class_start) * len(div_files[scout])),
                len(div_files[scout]) - 1,
            )
            div_files[scout].insert(index, hidden_sample)

    print("** Distributing index files")

    for i, host in enumerate(config.deploy.scouts):
        with NamedTemporaryFile(mode="w", delete=True) as fp:
            fp.write("\n".join(div_files[i]))
            fp.write("\n")

            print(f"- {len(div_files[i])} entries for {host}:{index_file}")
            if not dry_run:
                Connection(str(host)).put(fp.name, remote=index_file)
    return 0


def split_dataset(args: argparse.Namespace) -> int:
    config = load_config(args.mission_config)
    return shuffle_and_generate_index_files(
        config, args.split_by_prefix, args.random_seed, args.dry_run
    )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--dry-run", action="store_true")
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--split-by-prefix", action="store_true")
    parser.add_argument("mission_config", type=Path)
    args = parser.parse_args()

    ret = split_dataset(args)
    if ret:
        print("Failed to push new index files to scouts")

    sys.exit(ret)
