# SPDX-FileCopyrightText: 2023,2024 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

import shlex
from pathlib import Path
from typing import Optional

from fabric.exceptions import GroupException
from fabric.group import ThreadingGroup
from invoke.exceptions import UnexpectedExit

from ..deploy_config import DeployConfig


def _connect(config: DeployConfig) -> ThreadingGroup:
    return ThreadingGroup(*[str(scout) for scout in config.scouts])


def _start_hawk_scout(conn: ThreadingGroup, config: DeployConfig) -> None:
    cmd = shlex.join(
        [
            "tmux",
            "new-session",
            "-d",
            "hawk-venv/bin/hawk_scout",
            "--a2s-port",
            str(config.a2s_port),
        ],
    )
    conn.run(cmd, hide="out", echo=True, warn=True, disown=True)


def _stop_hawk_scout(conn: ThreadingGroup) -> None:
    conn.run("killall hawk_scout", hide="both", warn=True, echo=True)


def deploy(
    config: DeployConfig,
    dist_wheel: Path,
    dist_requirements: Optional[Path] = None,
) -> int:
    with _connect(config) as c:
        _stop_hawk_scout(c)

        # make sure venv exists
        cmd = shlex.join(
            [
                "python3",
                "-m",
                "venv",
                "hawk-venv",
            ]
        )
        c.run(cmd, hide="both", echo=True)

        # upload wheel
        c.put(dist_wheel)

        # uninstall
        try:
            cmd = shlex.join(
                [
                    "hawk-venv/bin/python",
                    "-m",
                    "pip",
                    "uninstall",
                    "--yes",
                    dist_wheel.name,
                ]
            )
            c.run(
                cmd,
                hide="both",
                echo=True,
            )
        except (GroupException, UnexpectedExit):
            pass

        # reinstall
        pip_install = [
            "hawk-venv/bin/python",
            "-m",
            "pip",
            "install",
        ]
        if dist_requirements is not None:
            c.put(dist_requirements, "requirements-scout.txt")
            pip_install.append("--constraint=requirements-scout.txt")

        # pip_install.append(f"hawk[scout] @ file://@HOME@/{dist_wheel.name}")
        pip_install.append(f"./{dist_wheel.name}[scout]")

        cmd = shlex.join(pip_install).replace("@HOME@", "'$HOME'")
        c.run(
            cmd,
            hide="out",
            echo=True,
            warn=True,
        )

        _start_hawk_scout(c, config)
    return 0


def restart_deployment(config: DeployConfig) -> int:
    """Stop previously deployed scouts."""
    with _connect(config) as c:
        _stop_hawk_scout(c)
        _start_hawk_scout(c, config)
    return 0


def stop_deployment(config: DeployConfig) -> int:
    """Stop previously deployed scouts."""
    with _connect(config) as c:
        _stop_hawk_scout(c)
    return 0


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["deploy", "restart", "stop"])
    parser.add_argument("config", type=Path)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()

    config = DeployConfig.from_yaml(args.config.read_text())

    if args.action == "deploy":
        ret = deploy(config, args.wheel)
    elif args.action == "restart":
        ret = restart_deployment(config)
    else:
        ret = stop_deployment(config)

    sys.exit(ret)
