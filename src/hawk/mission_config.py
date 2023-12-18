# SPDX-FileCopyrightText: 2023 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .deploy_config import DeployConfig


@dataclass
class MissionConfig:
    """Class representing a Hawk mission configuration."""

    config: dict[str, Any]
    deploy: DeployConfig

    # helpers to transition from dict to class
    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    @property
    def scouts(self) -> list[str]:
        if "scouts" in self.config:
            return [str(scout) for scout in self.config["scouts"]]
        return [scout.host for scout in self.deploy.scouts]


def load_config(config: Path) -> MissionConfig:
    config_dict = yaml.safe_load(config.read_text())

    if "deploy" not in config_dict:
        config_dict["deploy"] = dict(
            scouts=config_dict["scouts"],
            dataset="/srv/diamond",
        )
    deploy_config = DeployConfig.from_dict(config_dict)

    return MissionConfig(config=config_dict, deploy=deploy_config)


def write_config(config: MissionConfig, config_path: Path) -> None:
    with config_path.open("w") as fh:
        yaml.dump(config.config, fh)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    print(config)
