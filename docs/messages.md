<!--
SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>

SPDX-License-Identifier: GPL-2.0-only
-->


# Hawk Messages

## Introduction

List of protobuf messages used in Hawk RPC calls.
*   [MissionId](#MissionId)
*   [ScoutConfiguration](#ScoutConfiguration)
*   [MissionStats](#MissionStats)
*   [TestResults](#TestResults)
*   [MissionResults](#MissionResults)
*   [ModelArchive](#ModelArchive)
*   [TrainConfig](#TrainConfig)
*   [SVMConfig](#SVMConfig)
*   [SVMMode](#SVMMode)
*   [FastMPNCOVConfig](#FastMPNCOVConfig)
*   [FinetuneConfig](#FinetuneConfig)
*   [YOLOConfig](#YOLOConfig)
*   [RetrainPolicyConfig](#RetrainPolicyConfig)
*   [AbsolutePolicyConfig](#AbsolutePolicyConfig)
*   [PercentagePolicyConfig](#PercentagePolicyConfig)
*   [ModelPolicyConfig](#ModelPolicyConfig)
*   [Dataset](#Dataset)
*   [SelectiveConfig](#SelectiveConfig)
*   [TopKConfig](#TopKConfig)
*   [ThresholdConfig](#ThresholdConfig)
*   [PushPullConfig](#PushPullConfig)
*   [ReexaminationStrategyConfig](#ReexaminationStrategyConfig)
*   [TileMetadata](#TileMetadata)
*   [LabeledTile](#LabeledTile)
*   [LabelWrapper](#LabelWrapper)
*   [HawkObject](#HawkObject)
*   [SendTiles](#SendTiles)
*   [SendLabels](#SendLabels)

* * *

### Messages

### **MissionId**

Unique Mission Identifier

| Fields  | Type  | Description |
| ------------ |---------------| -----|
| value      | string | Identifier |


### **ScoutConfiguration**

Message to configure mission

| Fields  | Type  | Required | Description |
| --------|---- |---------------| -----|
| missionId    | string | True | Identifier |
| scouts | repeated string | True | Scout IP addresses|
| scoutIndex | int32 | True| Scout index |
| homeIP | string | True | Home IP address |
| trainLocation | string | True | Train location: scout / cloud / adaptive |
| missionDirectory| string | True | Path to directory to store models and logs | 
| trainStrategy| TrainConfig | True | Learning strategy configuration| 
| retrainPolicy| RetrainPolicyConfig| True| Retrain policy configuration |
| dataset| Dataset | True| Input data configuration|
| selector| SelectiveConfig | True| Selective tile transmission trategy|
| reexamination| ReexaminationStrategyConfig | True | Reexamination Strategy | 
| initialModel | ModelArchive | False| Initial model weight|
| bootstrapZip | bytes | True| Zip archive of bootstrap directory| 
| bandwidthFunc | map<int, string> | True| Maps network bandwidth over time (-1: end of mission)|

### **MissionStats**

Parameters in mission statistics

| Fields  | Type  | Description |
| ------------ |---------------| -----|
| totalObjects| int64| Total objects to be retrieved |
| processedObjects|int64| Objects processed |
| droppedObjects|int64| Corrupted images dropped|
| falseNegatives|int64| Positives not transmitted to labeler|
| others|map<string, string>| Miscellaneous statistics|

### **TestResults**

| Fields  | Type  | Description |
| ------------ |---------------| -----|
| testExamples | int64 | Number of examples in test dataset|
| auc| double | Model AUC|
| modelMetrics| ModelMetrics |Model metrics|
| bestThreshold| double |threshold at max F1 score|
| precisions| repeated | Precision at varing thresholds|
| recalls| repeated |Recall at varing thresholds |
| version | int32 |Model version|

### **MissionResults**

| Fields  | Type  | Description |
| ------------ |---------------| -----|
| results | map <int, TestResults> | Map of model metrics for all versions trained on test data|

### **ModelMetrics**

| Fields  | Type  | Description |
| ------------ |---------------| -----|
|truePositives| int64| True positives in validation dataset |
|falsePositives|int64| False Positives in validation dataset |
|falseNegatives|int64| False Negatives in validation dataset |
|precision|     double| Precision during validation|
|recall|        double| Recall during validation|
|f1Score|       double| F1Score during validation|

### **ModelArchive**

| Fields  | Type  | Required | Description |
| --------|---- |---------------| -----|
| arch | string| True | Model Architecture|
| content| bytes| True | Compressed model weight (zip)|


### **TrainConfig**

One of 

| Fields  | Type  | Description |
| ------------ |---------------| -----|
| svm|       SVMConfig| SVM configuration|
| fastMPNCOV|FastMPNCOVConfig| Fine-grained configuration|
| dnn|       FinetuneConfig| DNN only configuration|
| dnn_svm|   FinetuneConfig| DNN + SVM Configuration|
| yolo|      YOLOConfig| YOLO configuration|

### **SVMConfig**

| Fields  | Type | Required | Description |
| --------|---- |---------------| -----|
|mode| SVMMode| True | Type of svm training|
|featureExtractor| string| False |  Type of feature extractor dnn |
|probability|bool| False | Score as a probability|
|linearOnly|bool| False | Use Linear SVM|

### **SVMMode**

Choice of SVM Training
1. MASTER_ONLY 
2. DISTRIBUTED 
3. ENSEMBLE 

### **FastMPNCOVConfig**
| Fields  | Type  | Description |
| ------------ |---------------| -----|
| distributed | bool | If distributed training|
| freeze| int32| Number of DNN layer to freeze|

### **FinetuneConfig**

| Fields  | Type | Required | Description |
| --------|---- |---------------| -----|
| arch | string | True | Model Architecture|
| args| map<string, string> | False | DNN training parameters|

### **YOLOConfig**

| Fields  | Type  | Description |
| ------------ |---------------| -----|
| imageSize| int32| tile size|
| yolo_args| map<string, string>| YOLO training parameters|

### **RetrainPolicyConfig**

One of 
1. AbsolutePolicyConfig
2. PercentagePolicyConfig
3. ModelPolicyConfig


### **AbsolutePolicyConfig**

| Fields  | Type  | Required | Description |
| ---------|--- |---------------| -----|
| threshold| int32| True | Number of examples to retrain|
| onlyPositives| bool| False | Only positive examples included|

### **PercentagePolicyConfig**

| Fields  | Type | Required | Description |
| --------|---- |---------------| -----|
| threshold| int32| True | Percentage of example increment to retrain|
| onlyPositives| bool| False | Only positive examples included|

### **ModelPolicyConfig**
(EXMPERIMENTAL) Replace with a new model from admin

| Fields  | Type  | Description |
| ------------ |---------------| -----|
| path| string| Path to model generations|


### **Dataset**

One of 
1. FileDataset
2. LiveCameraDataset

### **SelectiveConfig**

One of:
1. TopKConfig 
2. ThresholdConfig 
3. PushPullConfig 


### **TopKConfig**

| Fields  | Type  | Required | Description |
| ------------ |--|-------------| -----|
| k | int32| True | Number of tiles selected|
| batchSize| int32| True | Number of tiles processed before selecting|

### **ThresholdConfig**

| Fields  | Type  | Required | Description |
| ------------ |----|-----------| -----|
|threshold | double| True | Threshold above which tiles transmitted|

### **PushPullConfig**

| Fields  | Type  | Required | Description |
| ------------ |---|------------| -----|
| k | int32| True | Number of tiles selected|
| batchSize| int32| True | Number of tiles processed before selecting|
| bandwidth | double| True | Fraction of bandwidth used for transmission|


### **ReexaminationStrategyConfig**
| Fields  | Type  | Required | Description |
| ------------ |-----|----------| -----|
| type | string | True | Type of reexamination (top/ none/ full/)|
| k | int32| True | Number of tiles selected|

### **TileMetadata**
| Fields  | Type  | Description |
| ------------ |---------------| -----|
|objectId | string| Tile identifier|
|label | LabelWrapper| Tile label|

### **LabeledTile**
| Fields  | Type  | Description |
| ------------ |---------------| -----|
|object | HawkObject| Tile content and metadata|
|label | LabelWrapper| Tile label |

### **LabelWrapper**
| Fields  | Type  | Description |
| ------------ |---------------| -----|
|objectId | string| Tile identifier|
|scoutIndex | int32 | Index of parent scout |
|imageLabel | string| Image label|
|boundingBoxes | repeated string| List of bounding boxes xywh format|

### **HawkObject**
| Fields  | Type  | Description |
| ------------ |---------------| -----|
|objectId | string| Tile identifier|
|content | bytes| Tile content|
|attributes | map<bytes, bytes>| Tile metadata|

### **SendTiles**
| Fields  | Type  | Description |
| ------------ |---------------| -----|
| objectId | string| Tile identifier|
| scoutIndex | int32 | Index of parent scout |
| score | double| Tile score|
| version | int32| Model Version|
| attributes | map<bytes, bytes>| Tile metadata|

### **SendLabels**
| Fields  | Type  | Description |
| ------------ |---------------| -----|
|label | LabelWrapper| Tile label |
