<!--
SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>

SPDX-License-Identifier: GPL-2.0-only
-->


# S2H (Scout to Home) API

## Introduction

API calls to transmit selected tiles from scout to home.  
Uses Publisher-Subscriber messaging. The network is bandwidth constricted using FireQos.

### API Calls

### **s2h_send_tiles**

Call to send selected tiles with metadata (objectId, parentScout) to home

Publish:

*   [SendTiles](messages.md#SendTiles)

