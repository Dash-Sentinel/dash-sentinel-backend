import os
from common import Color
from colordetect import get_image_color
from statistics import mode
from detect_vehicle import process_video
from sys import argv

if not argv[1]:
    print("Specify video file as arg")
    exit(1)

# Extract car colors from the video
if not os.path.exists('image_captures'):
    os.makedirs('image_captures')
image_captures = process_video(argv[1], 'image_captures')
image_colors = list(map(get_image_color,image_captures))
car_color = mode(image_colors)

for image_capture, image_color in zip(image_captures, image_colors):
    print(image_capture, ": ", image_color)