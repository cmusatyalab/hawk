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
    session.install(f".[{component}]", "pytest")
    session.run("pytest", f"tests/test_entrypoints_{component}.py")
