<!--
SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>

SPDX-License-Identifier: GPL-2.0-only
-->


# S2S (Scout to Scout) API

## Introduction

API calls from coordinator scout to scout to share knowledge of learning such as examples or models.  
Uses Request-Response messaging. The network can be bandwidth constrained with FireQOS (1Mbps in the paper).

List of calls in Hawk Admin API:
*   [s2s_configure_scout](#s2s_configure_scout)
*   [s2s_get_tile](#s2s_get_tile)
*   [s2s_add_tile_and_label](#s2s_add_tile_and_label)
*   [s2s_send_data](#s2s_send_data)

* * *

### API Calls

### **s2s_configure_scout**

Call to configure scout with user specified hyperparameters

Input:

*   [ScoutConfiguration](messages.md#ScoutConfiguration)

Output:

*   None

### **s2s_get_tile**

Call to get contents of requested tile ids

Input:

*   [TileMetadata](messages.md#TileMetadata)

Output:

*   [HawkObject](messages.md#HawkObject)



### **s2s_add_tile_and_label**

Call to transmit labeled image with metadata (labels / bounding box)

Input:

*   [LabeledTile](messages.md#LabeledTile)

Output:

*   None

### **s2s_send_data**

Call to send internal messages (bytes) between scouts

Input:

*   bytes

Output:

*   bytes