# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

from logzero import logger
from PIL import Image
from websockets.exceptions import ConnectionClosed
from websockets.server import WebSocketServerProtocol, serve


class UILabeler:
    def __init__(self, mission_dir: Path):
        self.config: dict[str, Any] = {}
        self._image_dir = mission_dir / "images"

        self.config["IMAGES"] = self._image_dir
        self.config["FILES"] = []
        self.config["LABELS"] = []
        self.config["HEAD"] = -1
        self.reload_directory()
        self.not_end = True

    def run(self, host: str, port: int) -> None:
        start_server = serve(self.transmit_results, host=host, port=port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

    async def transmit_results(self, websocket: WebSocketServerProtocol) -> None:
        size = (100, 100)
        while not len(self.config["FILES"]):
            time.sleep(5)
            self.reload_directory()
        while True:
            try:
                time.sleep(1)
                print("{} {}".format(len(self.config["FILES"]), self.config["HEAD"]))
                print(self.config["FILES"][0])
                image_path = self.config["FILES"][self.config["HEAD"]]
                logger.info(image_path)
                image = Image.open(image_path).convert("RGB")
                image.thumbnail(size, Image.LANCZOS)
                tmpfile = io.BytesIO()
                image.save(tmpfile, format="JPEG", quality=75)
                content = tmpfile.getvalue()
                encoded_img = base64.b64encode(content).decode("utf8")

                data = json.dumps(
                    {
                        "name": str(image_path),
                        "image": encoded_img,
                    }
                )
                await websocket.send(data)
                self.not_end = not (
                    self.config["HEAD"] == len(self.config["FILES"]) - 1
                )

                if self.not_end:
                    self.config["HEAD"] += 1
                else:
                    logger.info("Waiting for Results ...")
                    while not self.not_end:
                        time.sleep(5)
                        self.reload_directory()
            except ConnectionClosed:
                print("Client Disconnected !")
                break

    def reload_directory(self) -> None:
        old_length = len(self.config["FILES"])
        files = sorted(
            file for file in self.config["IMAGE"].iterdir() if file.is_file()
        )
        if not files:
            logger.error("No files")
            exit()
        new_length = len(files)
        new_files = new_length - old_length
        self.config["LABELS"].extend(["-1"] * new_files)
        self.config["FILES"] = files
        self.not_end = not (self.config["HEAD"] == new_length - 1)
        if new_files and self.config["HEAD"] < 0:
            self.config["HEAD"] = 0


def main() -> None:
    mission = Path(sys.argv[1])
    server = UILabeler(mission)
    server.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
