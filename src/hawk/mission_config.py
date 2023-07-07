# SPDX-FileCopyrightText: 2023 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .deploy_config import DeployConfig


@dataclass
class MissionConfig:
    """Class representing a Hawk mission configuration."""

    config: Dict[str, Any]
    deploy: Optional[DeployConfig]

    # helpers to transition from dict to class
    def __getitem__(self, key) -> Any:
        return self.config[key]

    def __setitem__(self, key, value) -> None:
        self.config[key] = value

    def get(self, key, default=None) -> Any:
        return self.config.get(key, default)

    @property
    def scouts(self) -> List[str]:
        if "scouts" in self.config:
            return self.config["scouts"]
        return [scout.host for scout in self.deploy.scouts]


def load_config(config: Path) -> MissionConfig:
    config_dict = yaml.safe_load(config.read_text())
    deploy_config = (
        DeployConfig.from_dict(config_dict) if "deploy" in config_dict else None
    )
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
