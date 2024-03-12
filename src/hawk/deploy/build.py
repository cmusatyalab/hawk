# SPDX-FileCopyrightText: 2023,2024 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

import subprocess
from pathlib import Path


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

    return Path(f"dist/{project}-{version}-py3-none-any.whl")


def poetry_export_requirements() -> Path:
    requirements_txt = Path("dist/requirements-scout.txt")
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


if __name__ == "__main__":
    print(poetry_build())
    print(poetry_export_requirements())
