# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

#!/usr/bin/env python3

from pathlib import Path
from subprocess import check_call

proto_root = Path.cwd() / 'protos'
proto_dir = proto_root / 'hawk' / 'proto'
for proto_file in proto_dir.iterdir():
    check_call(
        'protoc -I {} --python_out=. {}'
        .format(proto_root, proto_file)
        .split())

