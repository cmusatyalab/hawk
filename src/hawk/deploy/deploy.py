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
    cmd = [
        "tmux",
        "new-session",
        "-s",
        _session_name(config),
        "-d",
        "hawk-venv/bin/hawk_scout",
        "--a2s-port",
        str(config.a2s_port),
    ]
    conn.run(shlex.join(cmd), hide="out", echo=True, warn=True, disown=True)


def _stop_hawk_scout(conn: ThreadingGroup, config: DeployConfig) -> None:
    try:
        cmd = [
            "tmux",
            "kill-session",
            "-t",
            _session_name(config),
        ]
        conn.run(shlex.join(cmd), hide="both", echo=True, warn=True)
    except (GroupException, UnexpectedExit):
        pass


def _check_hawk_scout(conn: ThreadingGroup, config: DeployConfig) -> int:
    try:
        cmd = [
            "tmux",
            "has-session",
            "-t",
            _session_name(config),
        ]
        conn.run(shlex.join(cmd), hide="both", echo=True)
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
    *,
    install_uv: bool = False,
) -> int:
    PATH = "${HOME}/.local/bin/"
    with _connect(config) as c:
        if install_uv:
            print("Installing uv...")
            c.run("curl -LsSf https://astral.sh/uv/install.sh | sh", echo=True)

        # make sure venv exists
        cmd = ["uv", "venv", "--quiet", "--python", "3.10"]
        c.run(PATH + shlex.join(cmd), hide="out", echo=False)

        # upload wheel
        c.put(dist_wheel)

        _stop_hawk_scout(c, config)

        extra_indices = [
            "--index",
            "https://storage.cmusatyalab.org/wheels",
            "--index",
            "https://download.pytorch.org/whl/cu118",
            "--index-strategy",
            "unsafe-best-match",
        ]

        # uninstall
        if dist_requirements is not None:
            c.put(dist_requirements, "requirements-scout.txt")
            cmd = [
                "uv",
                "pip",
                "sync",
                "--quiet",
                *extra_indices,
                "requirements-scout.txt",
            ]
            c.run(PATH + shlex.join(cmd), hide="out", echo=True, warn=True)
        else:
            try:
                cmd = ["pip", "uninstall", "--quiet", dist_wheel.name]
                c.run(PATH + shlex.join(cmd), hide="both", echo=True)
            except (GroupException, UnexpectedExit):
                pass

        # reinstall
        cmd = ["uv", "pip", "install", "--quiet", *extra_indices]
        for extra_wheel in Path("wheels").glob("*.whl"):
            c.put(extra_wheel, extra_wheel.name)
            cmd.append(f"./{extra_wheel.name}")
        cmd.append(f"./{dist_wheel.name}")

        c.run(PATH + shlex.join(cmd), hide="out", echo=True, warn=True)

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
    parser.add_argument("--install-uv", action="store_true")
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
