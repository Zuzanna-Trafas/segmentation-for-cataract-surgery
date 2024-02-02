from transformers import AutoProcessor, AutoModelForUniversalSegmentation
from torch.utils.data import DataLoader
from cadis_dataset import CadisDataset

import argparse
import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



def create_custom_colormap(num_classes):
    if num_classes <= 20:
        #  use tab20 directly
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    elif num_classes <= 40:
        # combine tab20b and tab20c
        colors1 = plt.cm.tab20b(np.linspace(0, 1, 20))
        colors2 = plt.cm.tab20c(np.linspace(0, 1, 20))
        colors = np.vstack((colors1, colors2))
        colors = colors[np.linspace(0, colors.shape[0] - 1, num_classes).astype(int)]
    else:
        raise ValueError("Too many classes for this colormap generation method.")

    # Convert colors to 8-bit integers
    colormap = (colors[:, :3] * 255).astype(np.uint8)
    return colormap

def apply_custom_colormap(segmentation, colormap):
    # segmentation is 2D (H, W) instead of 3D (C, H, W)
    if segmentation.ndim == 2:
        segmentation = segmentation.unsqueeze(0)  # Add C dim

    # Prepare an RGB image with the same width and height as our segmentation mask
    height, width = segmentation.shape[1], segmentation.shape[2]
    colored_segmentation = np.zeros((height, width, 3), dtype=np.uint8)

    # Map each label to its corresponding color
    for label_index, color in enumerate(colormap):
        mask = segmentation[0] == label_index
        colored_segmentation[mask] = color

    return colored_segmentation

def save_segmentation_to_image(segmentation, filename, colormap):
    colored_segmentation = apply_custom_colormap(segmentation, colormap)
    segmentation_image = Image.fromarray(colored_segmentation)
    segmentation_image.save(filename)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_folder_name",
    help="Folder name where the model and processor are saved",
    default="oneformer_coco_swin_large_20240130_125029",
)
parser.add_argument("--output_dir", help="Directory to save segmented images", required=True)
args = parser.parse_args()

model_path = f"/home/data/cadis_results/trained_models/{args.model_folder_name}"

# Load the processor
processor_save_path = os.path.join(model_path, "processor")
processor = AutoProcessor.from_pretrained(processor_save_path)

# Load the model
model_save_path = os.path.join(model_path, "model")
model = AutoModelForUniversalSegmentation.from_pretrained(model_save_path, is_training=False)
model.eval()

video_numbers = [2]

# in CaDis, there are 35 possible labels
colormap = create_custom_colormap(35)
file_num = 1
for video_number in video_numbers:
    test_dataset = CadisDataset(processor, video_numbers=[video_number])
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for batch, original_image, ground_truth_labels in test_dataloader:

        # Drop the keys that are not accepted by the model in eval mode
        keys_to_drop = ["mask_labels", "class_labels", "text_inputs"]
        batch_ = {key: value for key, value in batch.items() if key not in keys_to_drop}

        with torch.no_grad():
            outputs = model(**batch_)

        # Postprocessing
        segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[[540, 960]])[0]
        print(type(segmentation))
        filename = os.path.join(args.output_dir, 'seg_' + str(file_num) + '.png')
        save_segmentation_to_image(segmentation, filename, colormap)
        file_num += 1
