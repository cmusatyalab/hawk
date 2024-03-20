# SPDX-FileCopyrightText: 2023,2024 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

import shlex
from pathlib import Path
from typing import Optional

from fabric.exceptions import GroupException
from fabric.group import ThreadingGroup
from invoke.exceptions import UnexpectedExit

from ..deploy_config import DeployConfig


def _session_name(config: DeployConfig) -> str:
    return f"hawk_scout_{config.a2s_port}"


def _connect(config: DeployConfig) -> ThreadingGroup:
    return ThreadingGroup(*[str(scout) for scout in config.scouts])


def _start_hawk_scout(conn: ThreadingGroup, config: DeployConfig) -> None:
    cmd = shlex.join(
        [
            "tmux",
            "new-session",
            "-s",
            _session_name(config),
            "-d",
            "hawk-venv/bin/hawk_scout",
            "--a2s-port",
            str(config.a2s_port),
        ],
    )
    conn.run(cmd, hide="out", echo=True, warn=True, disown=True)


def _stop_hawk_scout(conn: ThreadingGroup, config: DeployConfig) -> None:
    try:
        cmd = shlex.join(
            [
                "tmux",
                "kill-session",
                "-t",
                _session_name(config),
            ],
        )
        conn.run(cmd, hide="both", echo=True, warn=True)
    except (GroupException, UnexpectedExit):
        pass


def _check_hawk_scout(conn: ThreadingGroup, config: DeployConfig) -> int:
    try:
        cmd = shlex.join(
            [
                "tmux",
                "has-session",
                "-t",
                _session_name(config),
            ],
        )
        conn.run(cmd, hide="both", echo=True)
        return 0
    except (GroupException, UnexpectedExit):
        return 1


def check_deployment(config: DeployConfig) -> int:
    """Check deployed scouts."""
    with _connect(config) as c:
        return _check_hawk_scout(c, config)


def deploy(
    config: DeployConfig,
    dist_wheel: Path,
    dist_requirements: Optional[Path] = None,
) -> int:
    with _connect(config) as c:
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

        _stop_hawk_scout(c, config)

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
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu118",
        ]
        if dist_requirements is not None:
            c.put(dist_requirements, "requirements-scout.txt")
            pip_install.append("--constraint=requirements-scout.txt")

        # pip_install.append(f"cmuhawk[scout] @ file://@HOME@/{dist_wheel.name}")
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
        _stop_hawk_scout(c, config)
        _start_hawk_scout(c, config)
        return _check_hawk_scout(c, config)


def stop_deployment(config: DeployConfig) -> int:
    """Stop previously deployed scouts."""
    with _connect(config) as c:
        _stop_hawk_scout(c, config)
        return 1 if _check_hawk_scout(c, config) else 0


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "deploy", "restart", "stop"])
    parser.add_argument("config", type=Path)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()

    config = DeployConfig.from_yaml(args.config.read_text())

    if args.action == "check":
        ret = check_deployment(config)
        if ret:
            print("Not all scouts are deployed")
    elif args.action == "deploy":
        ret = deploy(config, args.wheel)
        if ret:
            print("Deployment failed")
    elif args.action == "restart":
        ret = restart_deployment(config)
        if ret:
            print("Failed to restart scouts")
    else:
        ret = stop_deployment(config)
        if ret:
            print("Failed to stop scouts")

    sys.exit(ret)
