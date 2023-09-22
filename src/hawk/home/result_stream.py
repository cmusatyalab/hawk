# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import asyncio
import base64
import io
import json
import os
import sys
import time
from os import walk

import websockets
from logzero import logger
from PIL import Image


class UILabeler:
    def __init__(self, mission_dir):
        self.config = {}
        mission_dir = str(mission_dir)
        self._image_dir = os.path.join(mission_dir, "images")
        self._meta_dir = os.path.join(mission_dir, "meta")
        self._label_dir = os.path.join(mission_dir, "labels")

        directory = self._image_dir
        if directory[len(directory) - 1] != "/":
            directory += "/"
        self.config["IMAGES"] = directory
        self.config["LABELS"] = []
        self.config["HEAD"] = -1
        self.config["FILES"] = []
        self.reload_directory()
        self.not_end = True

    def run(self, host, port):
        start_server = websockets.serve(self.transmit_results, host=host, port=port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

    async def transmit_results(self, websocket):
        size = (100, 100)
        while True:
            try:
                time.sleep(1)
                while not len(self.config["FILES"]):
                    time.sleep(5)
                    self.reload_directory()
                print("{} {}".format(len(self.config["FILES"]), self.config["HEAD"]))
                if len(self.config["FILES"]):
                    print(self.config["FILES"][0])
                image_name = self.config["FILES"][self.config["HEAD"]]
                image_name = os.path.join(self.config["IMAGES"], image_name)
                logger.info(image_name)
                image = Image.open(image_name).convert("RGB")
                image.thumbnail(size, Image.LANCZOS)
                content = io.BytesIO()
                image.save(content, format="JPEG", quality=75)
                content = content.getvalue()
                encoded_img = base64.b64encode(content).decode("utf8")

                data = json.dumps(
                    {
                        "name": image_name,
                        "image": encoded_img,
                    }
                )
                await websocket.send(data)
                self.not_end = not (
                    self.config["HEAD"] == len(self.config["FILES"]) - 1
                )

                if self.not_end:
                    self.config["HEAD"] = self.config["HEAD"] + 1
                else:
                    logger.info("Waiting for Results ...")
                    while not self.not_end:
                        time.sleep(5)
                        self.reload_directory()
            except websockets.exceptions.ConnectionClosed as e:
                print("Client Disconnected !")
                break

    def reload_directory(self):
        files = None
        old_length = len(self.config["FILES"])
        for dirpath, dirnames, filenames in walk(self.config["IMAGES"]):
            files = sorted(filenames)
            break
        if files == None:
            logger.error("No files")
            exit()
        self.config["FILES"] = files
        new_files = len(self.config["FILES"]) - old_length
        [self.config["LABELS"].append("-1") for i in range(new_files)]
        self.not_end = not (self.config["HEAD"] == len(self.config["FILES"]) - 1)
        if new_files and self.config["HEAD"] < 0:
            self.config["HEAD"] = 0
        return


def main():
    mission = sys.argv[1]
    server = UILabeler(mission)
    server.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
