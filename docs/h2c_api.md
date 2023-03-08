<!--
SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>

SPDX-License-Identifier: GPL-2.0-only
-->


# H2C (Home to Coordinator) API

## Introduction

API calls to send tile labels from home to coordinator scout.  
Uses PUSH/PULL messaging. The network is bandwidth constricted using FireQos.

List of calls in H2C API:
*   [h2c_send_labels](#h2c_send_labels)
* * *

### API Calls


### **h2c_send_labels**

Call to send labels and metadata (objectId, parentScout) to coordinator

Publish:

*   [SendLabels](messages.md#SendLabels)
