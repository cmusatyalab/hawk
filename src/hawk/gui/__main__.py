# SPDX-FileCopyrightText: 2024-2026 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import argparse
import os
from pathlib import Path

from streamlit.web import bootstrap

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--listen", default="localhost", help="address to listen on")
parser.add_argument("-p", "--port", type=int, default=8501, help="port to listen on")
parser.add_argument(
    "logdir",
    type=Path,
    default=Path.cwd(),
    help="directory with mission logs",
    nargs="?",
)


def main() -> None:
    args = parser.parse_args()

    os.environ["HAWK_MISSION_DIR"] = str(args.logdir.resolve())

    gui_dir = Path(__file__).parent
    entrypoint = gui_dir.joinpath("app.py")
    config = {
        "browser_gatherUsageStats": False,
        "client_toolbarMode": "viewer",
        "server_address": args.listen,
        "server_port": args.port,
        "server_headless": True,
        # Hawk look and feel, user config.toml theme settings still override this.
        "theme_base": str(gui_dir.joinpath("theme.toml")),
    }
    bootstrap.load_config_options(flag_options=config)
    bootstrap.run(str(entrypoint), False, [str(args.logdir)], config)


if __name__ == "__main__":
    main()
