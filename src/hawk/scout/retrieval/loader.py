# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from importlib_metadata import entry_points
from pydantic import ValidationError

if TYPE_CHECKING:
    from .retriever import Retriever


def load_retriever(retriever: str) -> type[Retriever]:
    try:
        retriever_ep = entry_points(group="cmuhawk.retrievers")[retriever]
    except KeyError:
        msg = f"Unknown retriever: {retriever}"
        raise ImportError(msg) from None

    try:
        retriever_cls: type[Retriever] = retriever_ep.load()
        return retriever_cls
    except ModuleNotFoundError as e:
        raise ImportError from e


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", action="store_true")
    parser.add_argument("retriever")
    parser.add_argument("config", nargs="*", metavar="config=value")
    args = parser.parse_args()

    config = {k: v for k, v in [setting.split("=", 1) for setting in args.config]}

    try:
        retriever_cls = load_retriever(args.retriever)

        if args.schema:
            schema = retriever_cls.config_class.model_json_schema()
            print(json.dumps(schema, indent=4))
        else:
            retriever_config = retriever_cls.validate_config(config)
            print(f"{retriever_config!r}")
    except ImportError:
        print(f'Unknown retriever: "{args.retriever}"')
    except ValidationError as e:
        print(e)
