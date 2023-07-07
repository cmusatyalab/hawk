# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py
from pathlib import Path
import os
from setuptools import setup, find_packages
from setuptools.command.install import install
from subprocess import check_call

with open('README.md') as f:
    readme = f.read()

with open('LICENSES/GPL-2.0-only.txt') as f:
    license = f.read()

SRC_PATH = os.path.relpath(os.path.dirname(__file__) or '.')


class PreInstallCommand(install):
    """Pre-installation for installation mode."""

    def run(self):
        proto_root = Path.cwd() / 'protos'
        proto_dir = proto_root / 'hawk' / 'proto'
        for proto_file in proto_dir.iterdir():
            check_call(
                'protoc -I {} --python_out=. {}'
                .format(proto_root, proto_file)
                .split())
        with open('hawk/proto/__init__.py', 'w') as fp:
            pass


        install.run(self)


setup(
    name='hawk',
    version='0.1.0',
    description='Hawk - Low-Bandwidth Remote Sensing of Rare Events',
    long_description=readme,
    author='Satya Lab',
    author_email='satya@cs.cmu.edu',
    url='https://github.com/cmusatyalab/hawk',
    license=license,
    cmdclass={
        'install': PreInstallCommand,
    },
    packages=find_packages(exclude=('tests', 'docs')),
    package_dir={
        "": SRC_PATH,
    },
    package_data={
        "": ['hawk/proto/*'],
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'hawk = hawk.scout.server_main:main',
        ]
    },
)
