from transformers import AutoProcessor, AutoModelForUniversalSegmentation
from torch.utils.data import DataLoader
from cadis_dataset import CadisDataset

import argparse
import torch
import os
from PIL import Image
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_folder_name",
    help="Folder name where the model and processor are saved",
    default="oneformer_coco_swin_large_20240130_125029",
)
parser.add_argument("--output_dir", help="Directory to save segmented images", default="/home/data/cadis_results/segmented_images")
args = parser.parse_args()

model_path = f"/home/data/cadis_results/trained_models/{args.model_folder_name}"

# Load the processor
processor_save_path = os.path.join(model_path, "processor")
processor = AutoProcessor.from_pretrained(processor_save_path)

# Load the model
model_save_path = os.path.join(model_path, "model")
model = AutoModelForUniversalSegmentation.from_pretrained(model_save_path, is_training=False)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for video_num in [2, 12, 22]:
    test_dataset = CadisDataset(processor, video_numbers=[video_num])
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for i, (batch, original_image, ground_truth_labels) in enumerate(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        # Drop the keys that are not accepted by the model in eval mode
        keys_to_drop = ["mask_labels", "class_labels", "text_inputs"]
        batch_ = {key: value for key, value in batch.items() if key not in keys_to_drop}

        with torch.no_grad():
            outputs = model(**batch_)

        # Postprocessing
        segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[[540, 960]])[0]
        filename = os.path.join(args.output_dir, f'seg_{video_num}_{i}.png')
        segmentation_image = Image.fromarray(segmentation.cpu().numpy().astype(np.uint8))
        segmentation_image.save(filename)