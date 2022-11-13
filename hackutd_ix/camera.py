from datetime import datetime
from time import sleep
import os
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from collections import deque
from common import Color
from colordetect import get_image_color
from statistics import mode
from detect_vehicle import process_video
#MAX_IMAGE_STORAGE = 10

# Record 

# Create a subdir to store the pictures if it doesn't exist
if not os.path.exists('image_captures'):
   os.makedirs('image_captures')

# use a queue to help in removing old images when max limit is reached
#filelist = deque()

#https://github.com/raspberrypi/picamera2/blob/main/examples/capture_mjpeg.py
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (1280, 720)})
picam2.configure(video_config)

encoder = H264Encoder(10000000)


while True:
    # Record the video
    video_filename = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + '.h264'
    picam2.start_recording(encoder, video_filename)
    sleep(5)
    picam2.stop_recording()
    print("Captured file: ", video_filename)

    # Extract car colors from the video
    if not os.path.exists('image_captures'):
        os.makedirs('image_captures')
    image_captures = process_video(video_filename, 'image_captures')
    image_colors = list(map(get_image_color,image_captures))
    car_color = mode(image_colors)

    for i in len(image_captures):
        print(image_captures[i], ": ", image_colors[i])

    # Clean up
    #os.remove(video_filename)
    #for image in image_captures:
    #    os.remove(image)
    #sleep(15)