from datetime import datetime
from time import sleep
import os
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from collections import deque

MAX_IMAGE_STORAGE = 10

# Record 

# Create a subdir to store the pictures if it doesn't exist
if not os.path.exists('image_captures'):
   os.makedirs('image_captures')

# use a queue to help in removing old images when max limit is reached
filelist = deque()

# https://github.com/raspberrypi/picamera2/blob/main/examples/capture_jpeg.py
#picam2 = Picamera2()
#config = picam2.create_still_configuration()
#picam2.configure(config)

#picam2.start()

#https://github.com/raspberrypi/picamera2/blob/main/examples/capture_mjpeg.py
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (1280, 720)})
picam2.configure(video_config)

encoder = H264Encoder(10000000)


while True:
    #filename = 'image_captures/' + datetime.utcnow().strftime("%Y%m%dT%H%M%S") + '.jpg'
    filename = 'image_captures/' + datetime.utcnow().strftime("%Y%m%dT%H%M%S") + '.h264'
    #picam2.capture_file(filename)
    picam2.start_recording(encoder, filename)
    sleep(5)
    picam2.stop_recording()
    print("Captured file: ", filename)
    filelist.append(filename)
    # Prune old files
    if len(filelist) > MAX_IMAGE_STORAGE:
        print("Removing image ", filelist[0])
        os.remove(filelist.popleft())
    sleep(10)