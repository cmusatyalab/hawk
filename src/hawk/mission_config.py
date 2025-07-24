# SPDX-FileCopyrightText: 2023 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

from dataclasses import asdict, dataclass
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
    def __contains__(self, key: str) -> bool:
        return key in self.config

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        return self.config.setdefault(key, default)

    @property
    def scouts(self) -> list[str]:
        if "scouts" in self.config:
            return [str(scout) for scout in self.config["scouts"]]
        return [scout.host for scout in self.deploy.scouts]

    def to_dict(self) -> dict[str, Any]:
        config = self.config.copy()
        config["deploy"] = asdict(self.deploy)
        return config

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> MissionConfig:
        # migrate config.scouts to config.deploy.scouts ('DeployConfig')
        if "deploy" not in config:
            config["deploy"] = {
                "scouts": config.get("scouts", []),
            }
        deploy_config = DeployConfig.from_dict(config)
        mission_config = {
            key: config[key] for key in config if key not in ["deploy", "scouts"]
        }
        return cls(
            config=mission_config,
            deploy=deploy_config,
        )

    @classmethod
    def from_yaml(cls, config: str) -> MissionConfig:
        return cls.from_dict(yaml.safe_load(config))


def load_config(config: Path) -> MissionConfig:
    return MissionConfig.from_yaml(config.read_text())


def write_config(config: MissionConfig, config_path: Path) -> None:
    with config_path.open("w") as fh:
        yaml.dump(config.to_dict(), fh)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    print(config)
