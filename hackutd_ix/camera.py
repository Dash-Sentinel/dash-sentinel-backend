from datetime import datetime
from time import sleep
import os
from picamera2 import Picamera2, Preview
from collections import deque

MAX_IMAGE_STORAGE = 10

# Create a subdir to store the pictures if it doesn't exist
if not os.path.exists('image_captures'):
   os.makedirs('image_captures')

# use a queue to help in removing old images when max limit is reached
filelist = deque()

# https://github.com/raspberrypi/picamera2/blob/main/examples/capture_jpeg.py
picam2 = Picamera2()
config = picam2.create_still_configuration()
picam2.configure(config)

picam2.start()

while True:
    filename = 'image_captures/' + datetime.utcnow().strftime("%Y%m%dT%H%M%S") + '.jpg'
    picam2.capture_file(filename)
    print("Captured file: ", filename)
    filelist.append(filename)
    # Prune old files
    if len(filelist) > MAX_IMAGE_STORAGE:
        print("Removing image ", filelist[0])
        os.remove(filelist.popleft())
    sleep(5)