<!--
SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>

SPDX-License-Identifier: GPL-2.0-only
-->


# H2A (Home to Admin) API

## Introduction

API calls from home to admin to transfer configuration file or models
Uses Request-Response messaging. The network is not constricted using FireQos.

List of calls in Admin API:
*   [h2a_send_config](#h2a_send_config)
*   [h2a_send_model](#h2a_send_model)
* * *

### API Calls

### **h2a_send_config**

Call to send content of configuration file

Input:

*  bytes

Output:

*   None

### **h2a_send_model**

Call to send path to model trained at home

Input:

*  string

Output:

*   None
