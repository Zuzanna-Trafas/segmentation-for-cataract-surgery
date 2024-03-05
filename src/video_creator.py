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
    "--every_n",
    help="Every n-th frame will be taken into the video",
    default=1,
    type=int
)
parser.add_argument(
    "--video_name",
    help="Name of the output video",
    default="output",
    type=str
)
parser.add_argument(
    "--start",
    help="Starting frame",
    default=0,
    type=int
)
parser.add_argument(
    "--end",
    help="Ending frame",
    default=None,
    type=int
)
args = parser.parse_args()
 
img_array = []
image_files = sorted(os.listdir(args.input_folder), key=natural_key)
if args.end is None:
    args.end = len(image_files)

for filename in image_files[args.start:args.end:args.every_n]:
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
