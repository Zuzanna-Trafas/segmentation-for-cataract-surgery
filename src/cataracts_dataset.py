from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import matplotlib.image as img
import os


from constants import CATARACTS_DATA_DIR
from utils.tools import natural_key


class CataractsDataset(Dataset):
    """
    Custom dataset class for semantic segmentation on the cataracts dataset (no labels but continuous frames).

    Args:
        processor: The processor for pre-processing images and labels.
        video_numbers (list): List of video numbers to include in the dataset.
        experiment (int): Experiment number for label remapping.
    """
    def __init__(self, processor, video_numbers=[2, 12, 22], fps=10, experiment=None):
        self.processor = processor
        self.video_numbers = video_numbers
        self.experiment = experiment
        self.fps = fps
        self.root_dir = CATARACTS_DATA_DIR
        self.load_metadata()

    def load_metadata(self):
        """Load metadata for the dataset."""
        self.video_metadata = []

        for video_number in self.video_numbers:
            formatted_video_number = f'{video_number:02d}'
            video_data = {'video_number': formatted_video_number, 'frames': []}

            images_path = os.path.join(self.root_dir, f'test{formatted_video_number}')

            frame_filenames = sorted(os.listdir(images_path), key=natural_key)

            every_n = 30 // self.fps

            for i in range(0, len(frame_filenames), every_n):
                frame_filename = frame_filenames[i]
                frame_path = os.path.join(images_path, frame_filename)
                if os.path.isfile(frame_path):
                    video_data['frames'].append({
                        'frame_path': frame_path
                    })

            self.video_metadata.append(video_data)
    
    def __getitem__(self, idx):
        """Return next frame by index"""
        for video_data in self.video_metadata:
            if idx < len(video_data['frames']):
                frame_data = video_data['frames'][idx]
                frame_path = frame_data['frame_path']

                frame_image = np.array(Image.open(frame_path).resize((960, 540)))

                if frame_image.shape != (540, 960, 3):
                    raise ValueError("Image dimensions are not (540, 960, 3)")

                # Map the image and label to the form needed by the model
                inputs = self.processor(images=frame_image, task_inputs=["panoptic"], return_tensors="pt")
                inputs = {k:v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}

                return inputs, frame_image, frame_path

            idx -= len(video_data['frames'])

    def __len__(self):
        total_frames = sum(len(video_data['frames']) for video_data in self.video_metadata)
        return total_frames
