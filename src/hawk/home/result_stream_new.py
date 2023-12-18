# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

import asyncio
import base64
import io
import json
import sys
from pathlib import Path
from typing import Set

from PIL import Image

# import aiofiles
from websockets.server import WebSocketServerProtocol, serve


class ResultStreamer:
    def __init__(self, mission_dir: Path):
        self._image_dir = mission_dir / "images"
        self._meta_dir = mission_dir / "meta"

    def run(self, host: str, port: int) -> None:
        start_server = serve(self.watch_directory, host=host, port=port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

    async def watch_directory(self, websocket: WebSocketServerProtocol) -> None:
        # Create a set to store the filenames of existing files in the directory
        # existing_files = set(os.listdir(self._image_dir))
        existing_files: Set[str] = set()
        size = (100, 100)

        while True:
            # Get the list of current files in the directory
            current_files = {
                file.name for file in self._image_dir.iterdir() if file.is_file()
            }

            # Find the new files by comparing the current files with the existing files
            new_files = current_files - existing_files

            # Process new files
            for filename in new_files:
                await asyncio.sleep(0.1)
                image_path = self._image_dir / filename
                # Check if the file is an image file
                if image_path.suffix in [".jpg", ".jpeg", ".png", ".gif"]:
                    # async with aiofiles.open(file_path, mode='rb') as file:
                    # Perform your image processing logic here
                    # Example: Read the file contents
                    # contents = await file.read()
                    if image_path.stat().st_size == 0:
                        print("Skipping empty file")
                        current_files.remove(filename)
                        continue
                    print(f"New image file '{image_path}' arrived!")
                    # --- Get the label to feed into flutter
                    image_number = str(image_path).split("/images/")[1].split(".")[0]
                    image_meta_path = self._meta_dir / f"{image_number}.json"
                    with open(image_meta_path) as f:
                        meta_data = json.load(f)
                    meta_orig_file = meta_data["objectId"]
                    label = meta_orig_file.split("/")[1]
                    # ---
                    image = Image.open(image_path).convert("RGB")
                    image.thumbnail(size, Image.LANCZOS)
                    tmpfile = io.BytesIO()
                    image.save(tmpfile, format="JPEG", quality=75)
                    content = tmpfile.getvalue()
                    encoded_img = base64.b64encode(content).decode("utf8")

                    data = json.dumps(
                        {
                            "name": image_path.name + " " + label,
                            "image": encoded_img,
                        }
                    )

                    # Serve the image over the WebSocket
                await websocket.send(data)

            # Update the set of existing files
            existing_files = current_files

            # Wait for a certain interval before checking again
            # print("Before sleeping...")
            await asyncio.sleep(1)
            # print("After sleeping...")

    """
    async def handle_websocket(websocket, path):
        # Specify the directory to monitor
        directory_path = '/path/to/your/directory'

        # Start watching the directory for new image files
        await watch_directory(directory_path, websocket)

    # Usage
    start_server = websockets.serve(handle_websocket, 'localhost', 8765)

    try:
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        pass
    """


def main() -> None:
    mission = Path(sys.argv[1])
    server = ResultStreamer(mission)
    server.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
