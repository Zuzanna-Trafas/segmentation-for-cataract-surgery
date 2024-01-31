from transformers import AutoProcessor, AutoModelForUniversalSegmentation
from torch.utils.data import DataLoader

import argparse
import torch
import os

from cadis_dataset import CadisDataset
from eval.metrics import calculate_mIoU
from utils.tools import save_segmentation


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_folder_name",
    help="Folder name where the model and processor are saved",
    default="oneformer_coco_swin_large_20240130_125029",
)
args = parser.parse_args()

model_path = f"/home/data/cadis_results/trained_models/{args.model_folder_name}"

# Load the processor
processor_save_path = os.path.join(model_path, "processor")
processor = AutoProcessor.from_pretrained(processor_save_path)

# Load the model
model_save_path = os.path.join(model_path, "model")
model = AutoModelForUniversalSegmentation.from_pretrained(
    model_save_path, is_training=False
)

model.eval()

test_dataset = CadisDataset(processor, video_numbers=[2])
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

data_iter = iter(test_dataloader)

# Get the first batch
batch, original_image, ground_truth_labels = next(data_iter)
ground_truth_labels = ground_truth_labels.squeeze(0)

# Drop the keys that are not accepted by the model in eval mode
keys_to_drop = ["mask_labels", "class_labels", "text_inputs"]
batch_ = {key: value for key, value in batch.items() if key not in keys_to_drop}

with torch.no_grad():
    outputs = model(**batch_)

# Postprocessing
segmentation = processor.post_process_semantic_segmentation(
    outputs, target_sizes=[[540, 960]]
)[0]

# Metrics
miou = calculate_mIoU(segmentation, ground_truth_labels)

save_segmentation(original_image, ground_truth_labels, segmentation, miou, processor, f"samples/sample_segmentation_{args.model_folder_name}.png")
