import cv2
import numpy as np
import glob
import os
import sys
import argparse

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
path = os.path.join(args.input_folder, '*.png')
image_files = sorted(glob.glob(path), key=lambda x: int(os.path.basename(x).split('_')[1]))
for filename in image_files:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
video_filepath = os.path.join(args.output_path, f"{args.video_name}.mp4")
out = cv2.VideoWriter(video_filepath,cv2.VideoWriter_fourcc(*'mp4v'), args.frame_rate, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
