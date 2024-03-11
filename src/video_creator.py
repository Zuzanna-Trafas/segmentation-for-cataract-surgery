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


def create_multiple_videos():
    for subdir in os.listdir(args.input_folder):
        subdir_path = os.path.join(args.input_folder, subdir)
        if os.path.isdir(subdir_path):
            video_filepath = os.path.join(args.output_path, f"{subdir}_fps_{args.frame_rate}.mp4")
            create_single_video(subdir_path, video_filepath)

def create_single_video(directory, output_path):
    img_array = []
    #image_files = sorted(os.listdir(directory), key=natural_key)
    #image_files = sorted(os.listdir(args.input_folder),key=lambda x: int(os.path.basename(x).split('_')[1]))
    image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and is_image_file(f)]
    image_files.sort(key=natural_key)
    if args.end is None:
        args.end = len(image_files)

    for filename in image_files[args.start:args.end:args.every_n]:
        img = cv2.imread(os.path.join(directory, filename))
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'), args.frame_rate, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print(f"Video created: {output_path}")

def has_subdirectories(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            return True
    return False

def is_image_file(filename):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']  # Add more extensions if needed
    return any(filename.endswith(extension) for extension in image_extensions)

# Example usage
if __name__ == "__main__":
    if has_subdirectories(args.input_folder):
        create_multiple_videos()
    else:
        video_filepath = os.path.join(args.output_path, f"{args.video_name}.mp4")
        create_single_video(args.input_folder, video_filepath)

    