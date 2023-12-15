from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch


# TODO change to load more videos
class CustomDataset(Dataset):
    def __init__(self, processor, video_number=1):
        self.processor = processor
        self.video_number = video_number
        self.first_frame = {
          1: 90,
          2: 430,
          3: 1550,
          4: 640,
          5: 300,
          6: 10,
          7: 400,
          8: 10,
          9: 1270,
        }

    def __getitem__(self, idx):
        dir = '/home/data/CaDISv2/Video01/'
        image = Image.open(f'{dir}/Images/Video{self.video_number}_frame{str(self.first_frame[self.video_number]+idx*10).zfill(6)}.png')
        map = Image.open(f'{dir}/Labels/Video{self.video_number}_frame{str(self.first_frame[self.video_number]+idx*10).zfill(6)}.png')
        map = np.array(map)

        # use processor to convert this to a list of binary masks, labels, text inputs and task inputs
        inputs = self.processor(images=image, segmentation_maps=map, task_inputs=["panoptic"], return_tensors="pt")
        inputs = {k:v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}

        return inputs

    def __len__(self):
        return 2
