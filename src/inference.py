from transformers import AutoProcessor, AutoModelForUniversalSegmentation
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import numpy as np
import argparse
import torch
import os

from custom_dataset import CustomDataset


parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    help="Path to the folder with saved model",
    default="/home/guests/zuzanna_trafas/segmentation-for-cataract-surgery/trained_models/oneformer_ade20k_swin_tiny_20231220_131920"
)
args = parser.parse_args()

# Load the processor
processor_save_path = os.path.join(args.path, "processor")  # Replace with the actual path
processor = AutoProcessor.from_pretrained(processor_save_path)

# Load the model
model_save_path = os.path.join(args.path, "model")  # Replace with the actual path
model = AutoModelForUniversalSegmentation.from_pretrained(model_save_path, is_training=False)

model.eval()

test_dataset = CustomDataset(processor, video_numbers=[2])
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

data_iter = iter(test_dataloader)

# Get the first batch
batch = next(data_iter)

# Drop the keys that are not accepted by the model in eval mode
keys_to_drop = ['mask_labels', 'class_labels', 'text_inputs']
batch = {key: value for key, value in batch.items() if key not in keys_to_drop}


# forward pass (no need for gradients at inference time)
with torch.no_grad():
    outputs = model(**batch)

# postprocessing
segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[[512, 910]])[0]


# draw the segmentatin
viridis = cm.get_cmap('viridis', torch.max(segmentation))
# get all the unique numbers
labels_ids = torch.unique(segmentation).tolist()
fig, ax = plt.subplots()
ax.imshow(segmentation)
handles = []
for label_id in labels_ids:
    label = processor.image_processor.metadata[str(label_id)]
    color = viridis(label_id)
    handles.append(mpatches.Patch(color=color, label=label))
ax.legend(handles=handles)
plt.savefig(f'samples/sample_segmentation_{args.path.split("/")[-1]}.png')
