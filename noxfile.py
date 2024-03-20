# SPDX-FileCopyrightText: 2023 Carnegie Mellon University
# SPDX-License-Identifier: 0BSD

# Test against different python versions
#
#   pipx install nox
#   pipx inject nox nox-poetry
#   nox

import nox
from nox_poetry import session


@session(python=["3.8", "3.9", "3.10"])
@nox.parametrize("component", ["home", "scout"])
def tests(session, component):
    session.install(
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu118",
        f".[{component}]",
        "pytest",
        "pytest-benchmark",
    )
    marker = "not scout" if component == "home" else "not home"
    session.run("pytest", "-m", marker)
