# SPDX-FileCopyrightText: 2023,2024 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import subprocess
from pathlib import Path

BuildError = Exception


def project_root() -> Path:
    for parent in Path("here").absolute().parents:
        if parent.joinpath("pyproject.toml").exists():
            return parent

    msg = "Unable to find root of the Hawk source tree"
    raise Exception(msg)


def poetry_build() -> Path:
    subprocess.run(
        [
            "poetry",
            "build",
            "--format=wheel",
            "--no-ansi",
        ],
        check=True,
    )

    result = subprocess.run(["poetry", "version"], capture_output=True, text=True)
    project, version = result.stdout.strip().split()

    root = project_root()
    return root / "dist" / f"{project}-{version}-py3-none-any.whl"


def poetry_export_requirements() -> Path:
    requirements_txt = project_root() / "dist" / "requirements-scout.txt"
    subprocess.run(
        [
            "poetry",
            "export",
            "--format=requirements.txt",
            "--only=main",
            "--extras=scout",
            "--without-hashes",
            "--output",
            str(requirements_txt),
        ],
        check=True,
    )
    return requirements_txt


def builder() -> tuple[Path, Path]:
    """build Hawk wheel and requirements files"""
    dist_wheel = poetry_build()
    if not dist_wheel.exists():
        msg = f"Could not find {dist_wheel}"
        raise BuildError(msg)

    dist_requirements = poetry_export_requirements()
    if not dist_requirements.exists():
        msg = f"Could not find {dist_requirements}"
        raise BuildError(msg)

    return dist_wheel, dist_requirements


if __name__ == "__main__":
    print(poetry_build())
    print(poetry_export_requirements())
