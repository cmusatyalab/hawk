<!--
SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>

SPDX-License-Identifier: GPL-2.0-only
-->



# A2S (Admin to Scout) API

## Introduction

API calls from admin to scouts to configure missions, explicitly start / stop mission, and other control calls.  
Uses Request-Response messaging. The network is not constricted using FireQos.

List of calls in Admin API:
*   [a2s_configure_scout](#a2s_configure_scout)
*   [a2s_start_mission](#a2s_start_mission)
*   [a2s_stop_mission](#a2s_stop_mission)
*   [a2s_get_mission_stats](#a2s_get_mission_stats)
*   [a2s_new_model](#a2s_new_model)
*   [a2s_get_test_results](#a2s_get_test_results)
*   [a2s_get_post_mission_archive](#a2s_get_post_mission_archive)
* * *

### API Calls

### **a2s_configure_scout**

Call to configure scout with user specified hyperparameters

Input:

*   [ScoutConfiguration](messages.md#ScoutConfiguration)

Output:

*   None   

### **a2s_start_mission**

Call to start mission including data retrieval, model processing, etc. on the scout

Input:

*   None

Output:

*   None   

### **a2s_stop_mission**

Call to stop mission on the scout

Input:

*   None

Output:

*   None

### **a2s_get_mission_stats**

Call to get statistics of mission from scout

Input:

*   None

Output:

*   [MissionStats](messages.md#MissionStats)

### **a2s_new_model**

 Call to transfer and load model on the scout.
 For experimental purposes. 

Input:

*   [ModelArchive](messages.md#ModelArchive)

Output:

*   None

### **a2s_get_test_results**

Call to get metrics of model on test data from scout

Input:

*   None

Output:

*   [MissionResults](messages.md#MissionResults)

### **a2s_get_post_mission_archive**

Call to get tar archive of logs and DNN models from scout

Input:

*   None

Output:

*   bytes
