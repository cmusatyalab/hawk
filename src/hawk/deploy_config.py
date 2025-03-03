# SPDX-FileCopyrightText: 2023 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cattrs
import yaml

from .ports import A2S_PORT, H2C_PORT, S2S_PORT

BANDWIDTH_PRESETS: Dict[str, Dict[str, int]] = {
    "12k": {"scout2client_speed": 12, "scout2client_delay": 2000},
    "30k": {"scout2client_speed": 30, "scout2client_delay": 2000},
    "100k": {"scout2client_speed": 100, "scout2client_delay": 1000},
    "1M": {"scout2client_speed": 1000, "scout2client_delay": 100},
}

SSH_DEFAULT_PORT = 22


@dataclass
class SshHost:
    """Class representing an SSH endpoint."""

    host: str
    user: Optional[str] = None
    port: int = SSH_DEFAULT_PORT

    def __str__(self) -> str:
        user = "" if self.user is None else f"{self.user}@"
        port = "" if self.port == SSH_DEFAULT_PORT else f":{self.port}"
        return f"{user}{self.host}{port}"

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def from_str(cls, ssh_host: str) -> "SshHost":
        user, host_port = ([None, *ssh_host.split("@", 1)])[-2:]
        assert host_port is not None
        host, port = ([*host_port.rsplit(":", 1), f"{SSH_DEFAULT_PORT}"])[:2]
        assert isinstance(host, str)
        return cls(host=host, user=user, port=int(port))


@dataclass
class Bandwidth:
    scout2client_speed: int  # speed in kbit/s
    scout2client_delay: int  # latency in ms
    scout2scout_speed: int = 1000  # speed in kbit/s


@dataclass
class DeployConfig:
    """Class representing the deployment part of the Hawk config file."""

    scouts: List[SshHost]
    scout_port: Optional[int] = None
    bandwidth: Optional[Bandwidth] = None

    @classmethod
    def from_yaml(cls, config_yaml: str) -> "DeployConfig":
        config_dict = yaml.safe_load(config_yaml)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DeployConfig":
        try:
            return cattrs.structure(config_dict["deploy"], cls)
        except Exception as exc:
            print(cattrs.transform_error(exc, path="deploy"))
            sys.exit(1)

    @property
    def a2s_port(self) -> int:
        return A2S_PORT if self.scout_port is None else self.scout_port

    @property
    def h2c_port(self) -> int:
        return H2C_PORT if self.scout_port is None else self.scout_port + 1

    @property
    def s2s_port(self) -> int:
        return S2S_PORT if self.scout_port is None else self.scout_port + 2

    @property
    def docker_image_name(self) -> str:
        return f"hawk_scout_{self.a2s_port}"


# to handle parsing bandwidth presets and yaml dict
cattrs.register_structure_hook(
    Bandwidth,
    lambda o, _: (
        Bandwidth(**BANDWIDTH_PRESETS[o])
        if isinstance(o, str) and o in BANDWIDTH_PRESETS
        else Bandwidth(**o)
    ),
)
# to handle parsing both "[user@]host[:port]" and yaml dict variants.
cattrs.register_structure_hook(
    SshHost, lambda o, _: SshHost.from_str(o) if isinstance(o, str) else SshHost(**o)
)
cattrs.register_structure_hook(Path, lambda o, _: Path(o))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    deploy_config = DeployConfig.from_yaml(args.config.read_text())
    print(deploy_config)
