# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator
from urllib.parse import urljoin
from zipfile import ZipFile

import requests
from logzero import logger

from ..hawkobject import MEDIA_SUFFIXES
from ..mission_config import MissionConfig


def unpack(bootstrap: ZipFile) -> Iterator[tuple[str, bytes, str]]:
    # extract positive bootstrap images from a bootstrap.zip
    for zipinfo in bootstrap.infolist():
        # skip anything in the top-level or negative examples directories
        name = Path(zipinfo.filename)
        if name.parent.name in ["", ".", "0"]:
            continue

        # skip non-image data
        content_type = MEDIA_SUFFIXES.get(name.suffix, "")
        if not content_type.startswith("image/"):
            continue

        with bootstrap.open(zipinfo, "r") as fd:
            content = fd.read()

        yield name.name, content, content_type


def resolve_scoped_dataset(config: MissionConfig) -> None:
    scope_config = config.get("scope", None)
    if scope_config is None:
        return

    ## scope config
    # scope:
    # *required fields*
    #   scope_server: http://cloudlet032.elijah.cs.cmu.edu:8000/
    #   embedding: clip-2
    #
    # *optional fields*
    #   limit: 500000
    #   class_list:
    #     - positive
    #   tiles_per_frame: 200
    #   chunksize: 10000
    #   shuffle: true
    #   shufflefirst: false
    #   seed: 0

    # required config
    scope_server = scope_config["scope_server"]
    embedding = scope_config["embedding"]
    bootstrap_zip = config["train_strategy"]["bootstrap_path"]

    # optional config
    limit = scope_config.get("limit", 500_000)
    class_list = scope_config.get("class_list", ["positive"])
    tiles_per_frame = scope_config.get("tiles_per_frame", 200)
    chunksize = scope_config.get("chunksize", 10_000)
    shuffle = scope_config.get("shuffle", True)
    shufflefirst = scope_config.get("shufflefirst", False)
    seed = scope_config.get("seed", 0)

    # get bootstrap images
    logger.info("Extracting positive bootstrap examples for scoping")
    with ZipFile(bootstrap_zip) as bootstrap:
        bootstrap_images = [
            ("images", (name, content, content_type))
            for name, content, content_type in unpack(bootstrap)
        ]

    # bootstrap_image -> clusters -> scoped webdataset descriptor
    with requests.Session() as s:
        logger.info(f"Finding nearby clusters for {len(bootstrap_images)} examples")
        find_clusters = urljoin(scope_server, f"/api/clusters/{embedding}")
        r = s.post(find_clusters, files=bootstrap_images)
        r.raise_for_status()
        clusters = r.json().get("clusters", [])

        # get scoped webdataset descriptor
        logger.info("Retrieving webdataset descriptor")
        create_scope = urljoin(
            scope_server, f"/clustered/{embedding}.json?limit={limit}"
        )
        r = s.post(create_scope, json=clusters)
        r.raise_for_status()
        wids = r.json()

    # create scoped dataset: config block
    logger.info("Creating scoped webdataset config")
    url_base = wids.get("base", "/")
    shardlist = [
        {"url": urljoin(url_base, shard["url"]), "nsamples": shard["nsamples"]}
        for shard in wids.get("shardlist", [])
    ]
    assert shardlist
    config.config["dataset"] = {
        "type": "webdataset",
        "shardlist": json.dumps(shardlist),
        "class_list": class_list,
        "tiles_per_frame": tiles_per_frame,
        "chunksize": chunksize,
        "seed": seed,
        "shuffle": shuffle,
        "shufflefirst": shufflefirst,
    }
