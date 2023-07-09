# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from subprocess import call

import numpy as np
import yaml

SPLIT="RANDOM"

def shuffle_and_generate_index_files(config):

    """
    Reads stream.txt from input dir and shuffles and
    sends index files to the hosts
    """
    data_config = config['dataset']
    stream_file = Path(data_config['stream_path'])
    index_file = data_config['index_path']
    hosts = config['scouts']
    seed = config.get('seed', None)
    assert stream_file.exists(), "Stream file does not exist"

    contents = sorted(open(stream_file).read().splitlines())

    img_tile_map = defaultdict(list)
    img_label_map = defaultdict(list)

    # Shuffle and split tiles
    random.seed(seed)
    random.shuffle(contents)
    total_tiles = len(contents)

    if SPLIT == "KEY":
        keys = set()
        for content in contents:
            path, label = content.split()
            k = os.path.basename(path).split('_')[0]
            keys.add(k)
    else:
        num_tiles_per_frame = data_config.get('tiles_per_frame', 200)
        num_frames = math.ceil(total_tiles/num_tiles_per_frame)
        keys = np.arange(num_frames)
        
    keys = list(keys)
    total_keys = len(keys)
    tiles_per_frame = math.ceil(total_tiles/total_keys)
    per_frame = np.array_split(contents, total_keys)
    print(len(per_frame[0]))

    for i, tiles_per_frame in enumerate(per_frame):
        k = keys[i]
        for content in tiles_per_frame:
            path, label = content.split()
            img_tile_map[k].append(content)
            img_label_map[k].append(int(label))

    keys = list(img_tile_map.keys())
    random.shuffle(keys)
    
    num_hosts = len(hosts)

    div_keys = [keys[i::num_hosts] for i in range(num_hosts)]

    div_files = []

    for keys in div_keys:
        paths = []
        for k in keys:
            files = img_tile_map[k]
            for f in files:
                paths.append(f)

        div_files.append(paths)

    dest_path = os.path.join(index_file)
    filename = os.path.basename(index_file)
    src_path = "/tmp/{}".format(filename)
    for i, host in enumerate(hosts):
        with open(src_path, "w") as f:
            f.write("\n".join(div_files[i]))
        # scp to hosts
        cmd = "scp {} root@{}:{}".format(src_path, host, dest_path)
        call(cmd.split(" "))
    return

if __name__ == "__main__": 
    config_path = sys.argv[1] if len(sys.argv) > 1 \
                else (Path.cwd() / 'configs/config.yml')
    random_seed = int(sys.argv[2]) if len(sys.argv) > 2 \
                else None

    with open(config_path) as f:
        config = yaml.safe_load(f)
    config['seed'] = random_seed

    shuffle_and_generate_index_files(config)
