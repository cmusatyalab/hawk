# SPDX-FileCopyrightText: 2023,2024 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import subprocess
from pathlib import Path

BuildError = Exception


def project_root() -> Path:
    for parent in Path("here").resolve().parents:
        if parent.joinpath("pyproject.toml").exists():
            return parent

    msg = "Unable to find root of the Hawk source tree"
    raise Exception(msg)


def build() -> Path:
    root = project_root()

    subprocess.run(["uv", "build", "--wheel"], check=True)

    result = subprocess.run(["uv", "version"], capture_output=True, text=True)
    project, version = result.stdout.strip().split()

    return root / "dist" / f"{project}-{version}-py3-none-any.whl"


def export_requirements() -> Path:
    requirements_txt = project_root() / "dist" / "requirements-scout.txt"
    subprocess.run(
        [
            "uv",
            "export",
            "--format",
            "requirements.txt",
            "--quiet",
            "--no-emit-project",
            "--no-dev",
            "--no-hashes",
            "--no-annotate",
            "--extra",
            "scout",
            "-o",
            str(requirements_txt),
        ],
        check=True,
    )
    return requirements_txt


def builder() -> tuple[Path, Path]:
    """build Hawk wheel and requirements files"""
    dist_wheel = build()
    if not dist_wheel.exists():
        msg = f"Could not find {dist_wheel}"
        raise BuildError(msg)

    dist_requirements = export_requirements()
    if not dist_requirements.exists():
        msg = f"Could not find {dist_requirements}"
        raise BuildError(msg)

    return dist_wheel, dist_requirements


if __name__ == "__main__":
    print(build())
    print(export_requirements())
