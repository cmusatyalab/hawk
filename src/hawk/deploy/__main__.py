# SPDX-FileCopyrightText: 2023,2024 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..deploy_config import DeployConfig
from .build import builder
from .dataset import split_dataset
from .deploy import check_deployment, deploy, restart_deployment, stop_deployment


def config_from_args(args: argparse.Namespace) -> DeployConfig:
    config = DeployConfig.from_yaml(args.config.read_text())
    print(f"{args.action} hawk-scout @ {config.scouts}")
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="action")

    def build_wrapper(args: argparse.Namespace) -> int:
        dist_wheel, dist_requirements = builder()
        print("built wheel and requirements files")
        print(f"{dist_wheel}\n{dist_requirements}")
        return 0

    build_parser = subparsers.add_parser(
        "build",
        help="Build cmuhawk wheel and requirements files",
    )
    build_parser.set_defaults(func=build_wrapper)

    def deploy_wrapper(args: argparse.Namespace) -> int:
        config = config_from_args(args)
        dist_wheel, dist_requirements = builder()
        print(f"deploying hawk-scout @ {config.scouts}")
        return deploy(config, dist_wheel, dist_requirements, install_uv=args.install_uv)

    deploy_parser = subparsers.add_parser("deploy", help="Deploy new Hawk scouts")
    deploy_parser.add_argument("--install-uv", action="store_true")
    deploy_parser.add_argument("config", type=Path)
    deploy_parser.set_defaults(func=deploy_wrapper)

    def check_wrapper(args: argparse.Namespace) -> int:
        config = config_from_args(args)
        ret = check_deployment(config)
        if ret:
            print("Not all deployed scouts are running")
        return ret

    check_parser = subparsers.add_parser(
        "check",
        help="Check if all deployed Hawk scouts are running",
    )
    check_parser.add_argument("config", type=Path)
    check_parser.set_defaults(func=check_wrapper)

    def restart_wrapper(args: argparse.Namespace) -> int:
        config = config_from_args(args)
        return restart_deployment(config)

    restart_parser = subparsers.add_parser(
        "restart",
        help="Restart deployed Hawk scouts",
    )
    restart_parser.add_argument("config", type=Path)
    restart_parser.set_defaults(func=restart_wrapper)

    def stop_wrapper(args: argparse.Namespace) -> int:
        config = config_from_args(args)
        return stop_deployment(config)

    stop_parser = subparsers.add_parser("stop", help="Stop deployed Hawk scouts")
    stop_parser.add_argument("config", type=Path)
    stop_parser.set_defaults(func=stop_wrapper)

    split_parser = subparsers.add_parser(
        "split",
        help="Split stream.txt (index) across deployed Hawk scouts",
    )
    split_parser.add_argument("-n", "--dry-run", action="store_true")
    split_parser.add_argument("--random-seed", type=int)
    split_parser.add_argument("--split-by-prefix", action="store_true")
    split_parser.add_argument("mission_config", type=Path)
    split_parser.set_defaults(func=split_dataset)

    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
