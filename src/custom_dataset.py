from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import torch


class CustomDataset(Dataset):
    def __init__(self, processor, video_numbers=[1]):
        self.processor = processor
        self.video_numbers = video_numbers
        self.root_dir = "/home/data/CaDISv2"
        self.load_metadata()

    def load_metadata(self):
        self.video_metadata = []

        for video_number in self.video_numbers:
            formatted_video_number = f'{video_number:02d}'
            video_data = {'video_number': formatted_video_number, 'frames': []}

            images_path = os.path.join(self.root_dir, f'Video{formatted_video_number}', 'Images')
            labels_path = os.path.join(self.root_dir, f'Video{formatted_video_number}', 'Labels')

            frame_filenames = os.listdir(images_path)

            for frame_filename in frame_filenames:
                frame_path = os.path.join(images_path, frame_filename)
                label_path = os.path.join(labels_path, frame_filename)

                if os.path.isfile(frame_path) and os.path.isfile(label_path):
                    video_data['frames'].append({
                        'frame_path': frame_path,
                        'label_path': label_path
                    })

            self.video_metadata.append(video_data)

    def __getitem__(self, idx):
        for video_data in self.video_metadata:
            if idx < len(video_data['frames']):
                frame_data = video_data['frames'][idx]
                frame_path, label_path = frame_data['frame_path'], frame_data['label_path']

                frame_image = np.array(Image.open(frame_path))
                label_image = np.array(Image.open(label_path))

                inputs = self.processor(images=frame_image, segmentation_maps=label_image, task_inputs=["panoptic"], return_tensors="pt")
                inputs = {k:v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}

                return inputs

            idx -= len(video_data['frames'])

    def __len__(self):
        total_frames = sum(len(video_data['frames']) for video_data in self.video_metadata)
        return total_frames
