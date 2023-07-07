# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

import asyncio
import base64
import json 
import io 
import os
import sys
import time
import websockets

from logzero import logger
from os import walk
from PIL import Image

import asyncio
import os
#import aiofiles
import websockets

class UILabeler:

    def __init__(self, mission_dir):
        self.config = {}
        mission_dir = str(mission_dir)
        self._image_dir = os.path.join(mission_dir, 'images')
        self._meta_dir = os.path.join(mission_dir, 'meta')
        self._label_dir = os.path.join(mission_dir, 'labels')
        
        directory = self._image_dir
        if directory[len(directory) - 1] != "/":
             directory += "/"
        self.config["IMAGES"] = directory
        self.config["LABELS"] = []
        self.config["HEAD"] = -1
        self.config["FILES"] = []
        #self.reload_directory()
        self.not_end = True 
        self.total_images_sent = 0

    def run(self, host, port):
        start_server = websockets.serve(self.watch_directory, host=host, port=port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

    async def watch_directory(self, websocket):
        # Create a set to store the filenames of existing files in the directory
        #existing_files = set(os.listdir(self._image_dir))
        existing_files = set()
        size = (100, 100)

        while True:
            # Get the list of current files in the directory
            current_files = set(os.listdir(self._image_dir))

            # Find the new files by comparing the current files with the existing files
            new_files = current_files - existing_files

            # Process new files
            for filename in new_files:
                await asyncio.sleep(.1)
                # Check if the file is an image file
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    image_name = os.path.join(self._image_dir, filename)
                    #async with aiofiles.open(file_path, mode='rb') as file:
                        # Perform your image processing logic here
                        # Example: Read the file contents
                        #contents = await file.read()
                    if os.stat(image_name).st_size == 0:
                        print("Skipping empty file")
                        current_files.remove(filename)
                        continue
                    print(f"New image file '{image_name}' arrived!")
                    ### Get the label to feed into flutter
                    image_number = image_name.split("/images/")[1].split('.')[0]
                    image_meta = image_number + ".json"
                    image_meta_path = os.path.join(self._meta_dir, image_meta)
                    with open(image_meta_path, "r") as f:
                        meta_data = json.load(f)
                    meta_orig_file = meta_data['objectId']
                    label = meta_orig_file.split("/")[1]
                    ###
                    image = Image.open(image_name).convert('RGB')
                    image.thumbnail(size, Image.LANCZOS)
                    content = io.BytesIO()
                    image.save(content, format='JPEG', quality=75)
                    content = content.getvalue()
                    encoded_img = base64.b64encode(content).decode('utf8')

                    data = json.dumps({'name': image_name + " " + label,
                                       'image': encoded_img,
                                       })

                        # Serve the image over the WebSocket
                await websocket.send(data)

            # Update the set of existing files
            existing_files = current_files

            # Wait for a certain interval before checking again
            #print("Before sleeping...")
            await asyncio.sleep(1)
            #print("After sleeping...")
    '''
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
    '''
def main():
    mission = sys.argv[1]
    server = UILabeler(mission)
    server.run(host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()