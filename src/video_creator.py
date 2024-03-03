import cv2
import numpy as np
import glob
import os
import sys
import argparse
import re


def natural_key(string_):
    """See https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_folder",
    help="Input folder that contains all the png pictures to be concatenated into a video",
    type=str
)
parser.add_argument(
    "--output_path",
    help="Output path where the video should be produced",
    default="",
    type=str
)
parser.add_argument(
    "--frame_rate",
    default=10,
    type=int
)
parser.add_argument(
    "--video_name",
    help="Name of the output video",
    default="output",
    type=str
)
args = parser.parse_args()
 
img_array = []
image_files = sorted(os.listdir(args.input_folder), key=natural_key)
for filename in image_files:
    img = cv2.imread(os.path.join(args.input_folder, filename))
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
video_filepath = os.path.join(args.output_path, f"{args.video_name}.mp4")
out = cv2.VideoWriter(video_filepath,cv2.VideoWriter_fourcc(*'mp4v'), args.frame_rate, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

print("Finish!!")
