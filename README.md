<!--
SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>

SPDX-License-Identifier: GPL-2.0-only
-->


# Hawk: Low-Bandwidth Remote Sensing of Rare Events

*Hawk* is a bandwidth-frugal learning system for discovering instances of rare
targets from weakly connected distributed data source called "scouts." The mission 
starts with a machine learning (ML) model trained on very few labeled data. 
The incoming data is inferred using the model. Hawk uses active learning techniques
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
python -m hawk.scout.server_main
```

### Step 5. Start Hawk at Home 

```bash
python -m hawk.home.home_main configs/config.yml
```
## Running Hawk UI
Hawk UI is developed using [Flutter SDK](https://docs.flutter.dev/get-started/install) and has been tested using Chrome browser.
The backend uses Flask REST API (Port:8000) to start and stop missions. The results are streamed to the front-end using websockets (Port:5000). 

### Step 1. Setting up Environment
```bash
cd ~/hawk
export PYTHONPATH=$PWD
conda env create -n <VIRTUAL_ENV_NAME> -f environment.yml
conda activate <VIRTUAL_ENV_NAME>
python scripts/generate_proto.py
```
### Step 2. ScopeCookie and Config

Assumes scope cookie (NEWSCOPE) and config file (config.yml) are present in ~/.hawk

```bash
cp ~/hawk/configs/flutter.yml ~/.hawk/config.yml
```
### Step 3. Start Home process
```bash
python -m hawk.home.home_flutter
```

### Step 4. Start Flutter app
```bash
cd ~/hawk/hawk_ui
export PYTHONPATH=$PWD
flutter run -d chrome
# if port forwarding use cmd below
# flutter run -d web-server --web-port=35415
```
Configure the filter using the UI and then click 'Start' to begin mission. The results can be viewed on 'Results' panel.

# Licensing

Unless otherwise stated, the source code is copyright Carnegie Mellon University and licensed under the GNU General Public License, version 2. Portions from the following third party sources have been modified and are included in this repository. These portions are noted in the source files and are copyrighted by their respective authors with the licenses listed.

Project | Modified | License
---|---|---|
[pytorch/examples](https://github.com/pytorch/examples) | Yes | BSD
[yolov5](https://github.com/ultralytics/yolov5) | Yes | GNUv3

