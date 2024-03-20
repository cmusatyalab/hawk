# Copyright (c) 2024 Carnegie Mellon University
# SPDX-License-Identifier: GPLv2-only

collect_ignore = []

# skip home tests when we only have scout specific dependencies installed
try:
    import fabric  # noqa: F401
except ImportError:
    collect_ignore.append("home")

# skip scout tests when we only have home specific dependencies installed
try:
    import torch  # noqa: F401
except ImportError:
    collect_ignore.append("scout")
