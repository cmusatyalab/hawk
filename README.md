<!--
SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>

SPDX-License-Identifier: GPL-2.0-only
-->


# Hawk: Low-Bandwidth Remote Sensing of Rare Events

*Hawk* is a bandwidth-frugal learning system for discovering instances of rare
targets from weakly connected distributed data source called "scouts." The mission 
starts with a machine learning (ML) model trained on very few labeled data. 
The incoming data is inferred using the model. Hawk uses acctive learning techniques 
to choose a small subset of data to query the labeler for annotations.
The obtained labels are added to the labeled training set to periodically improve the quality of 
models present in the scouts.

More details about can be found [here.](/docs/README.md) 

This is a developing project.

## Installation steps

### Step 1. Setting up Environment

```bash
cd ~/hawk
export PYTHONPATH=$PWD
conda env create -n <VIRTUAL_ENV_NAME> -f environment.yml
conda activate <VIRTUAL_ENV_NAME>
python setup.py install
```

### Step 2. Modify mission configuration 

Provide relevant data and model parameters in the [config file.](/home/configs/config.yml)  

### Step 3. Split dataset across Scout Servers 

The index file provided in [config file.](/home/configs/config.yml) contains list of input images.
Run the following cmd to split the dataset across participating scouts
```bash
cd ~/hawk
python scripts/split_data.py configs/config.yml
```


### Step 4. Start Hawk on the Scout Servers 

```bash
cd ~/hawk/hawk
python server_main.py
```

### Step 5. Start Hawk at Home 

```bash
cd ~/hawk/home
python home_main.py configs/config.yml
```

# Licensing

Unless otherwise stated, the source code copyright Carnegie Mellon University and licensed under the GNU General Public License, version 2. Portions from the following third party sources have been modified and are included in this repository. These portions are noted in the source files and are copyright their respective authors with the licenses listed.

Project | Modified | License
---|---|---|
[pytorch/examples](https://github.com/pytorch/examples) | Yes | BSD
[yolov5](https://github.com/ultralytics/yolov5) | Yes | GNUv3

