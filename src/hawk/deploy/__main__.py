# SPDX-FileCopyrightText: 2023,2024 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

import argparse
import sys
from pathlib import Path

from ..deploy_config import DeployConfig
from .build import poetry_build, poetry_export_requirements
from .deploy import deploy, restart_deployment, stop_deployment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["deploy", "restart", "stop"])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config = DeployConfig.from_yaml(args.config.read_text())

    if args.action == "restart":
        # restart remote (scout) containers.
        print(f"restarting hawk-scout @ {config.scouts}")
        return restart_deployment(config)

    if args.action == "stop":
        # stop remote (scout) containers.
        print(f"stopping hawk-scout @ {config.scouts}")
        return stop_deployment(config)

    # build Hawk wheel
    dist_wheel = poetry_build()
    if not dist_wheel.exists():
        print(f"Could not find {dist_wheel}")
        return 1

    # export requirements-scout.txt
    dist_requirements = poetry_export_requirements()
    if not dist_requirements.exists():
        print(f"Could not find {dist_requirements}")
        return 1

    print(f"deploying hawk-scout @ {config.scouts}")
    return deploy(config, dist_wheel, dist_requirements)


if __name__ == "__main__":
    sys.exit(main())
