from transformers import AutoProcessor, AutoModelForUniversalSegmentation
from torch.utils.data import DataLoader

import argparse
import torch
import os

from cadis_dataset import CadisDataset
from eval.evaluation import test


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

test_dataset = CadisDataset(processor, video_numbers=[2,12,22])
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

mIoU, mIoU_anatomy, mIoU_instruments, panoptic_quality, pac = test(
                model, processor, test_dataloader, device
            )

print(f"mIoU: {mIoU}")
print(f"mIoU_anatomy: {mIoU_anatomy}")
print(f"mIoU_instruments: {mIoU_instruments}")
print(f"panoptic_quality: {panoptic_quality}")  
print(f"pac: {pac}") 
