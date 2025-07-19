# SPDX-FileCopyrightText: 2023 Carnegie Mellon University
# SPDX-License-Identifier: 0BSD

# Test against different python versions
#
#   pipx install nox[uv]
#   nox

import nox

nox.options.default_venv_backend = "uv"


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12"])
@nox.parametrize("component", ["home", "scout"])
def tests(session, component):
    session.install(
        "--extra-index-url",
        "https://storage.cmusatyalab.org/wheels",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu118",
        "--index-strategy",
        "unsafe-best-match",
        f".[{component}]",
        "pytest",
        "pytest-benchmark",
    )
    marker = "not scout" if component == "home" else "not home"
    session.run("pytest", "-m", marker)
