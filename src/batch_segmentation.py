from transformers import AutoProcessor, AutoModelForUniversalSegmentation
from torch.utils.data import DataLoader
from PIL import Image
import argparse
import torch
import os
import time
import wandb
import numpy as np
import matplotlib.pyplot as plt

from cataracts_dataset import CataractsDataset
from utils.tools import apply_custom_colormap
from utils.cadis_visualization import get_cadis_colormap
from pupil_size_calculator import PupilSizeCalculator

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_folder_name",
    help="Folder name where the model and processor are saved",
    default="oneformer_coco_swin_large_20240130_125029",
)
#parser.add_argument("--output_dir", help="Directory to save segmented images", default="/home/data/cadis_results/segmentation_results")
parser.add_argument("--output_dir", help="Directory to save segmented images", default="/home/guests/dominika_darabos/segmentation-for-cataract-surgery/samples/video_sample")
args = parser.parse_args()

wandb.login()

config = {
    "model": args.model_folder_name,
    "output_dir": args.output_dir,
}

run = wandb.init(
    project="batch-segmentation",
    config=config,
)

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

colormap = get_cadis_colormap()

for video_num in [1]:
    test_dataset = CataractsDataset(processor, video_numbers=[video_num])
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    frame_number = len(test_dataloader)
    wandb.log({"frame_number": frame_number})
    for i, (batch, original_image, frame_path) in enumerate(test_dataloader):
        # Filter out black images
        if (np.unique(original_image) == [0]).all():
            continue
        start = time.time()
        batch_ = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch_)

        # Postprocessing
        segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[[540, 960]])[0]

        # Create a filename to save segmentation
        frame = frame_path[0].split("/")[-1]
        frame = frame.split(".")[0]
        filename = os.path.join(args.output_dir, f'test{video_num:02d}', f'{frame}.png')
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        segmentation = segmentation.cpu().detach()
        # Save segmentation result
        segmentation_image = Image.fromarray(segmentation.numpy().astype(np.uint8))
        segmentation_image.save(filename)

        calculator = PupilSizeCalculator(threshold=30)
        pupil_width = calculator.calculate_width(segmentation.numpy().astype(np.uint8))

        # Save colored segmentation alongside video frame
        filename_plot = os.path.join(args.output_dir, f'comparison_{video_num:02d}', f'{frame}.png')
        os.makedirs(os.path.dirname(filename_plot), exist_ok=True)
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        # Video frame
        original_image_squeezed = original_image.squeeze(0)
        axs[0].imshow(original_image_squeezed)
        axs[0].set_title("Video Frame")
        axs[0].axis("off")

        # Segmentation
        segmentation_squeezed = segmentation.squeeze(0)
        segmentation_squeezed = apply_custom_colormap(segmentation_squeezed, colormap)
        axs[1].imshow(segmentation_squeezed)
        axs[1].set_title("Segmented Image")
        axs[1].axis("off")
        if pupil_width is None:
            pupil_width = "invalid"
        axs[1].text(0.5, -0.1, f"Pupil width: {str(pupil_width)}", horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        plt.tight_layout()
        plt.savefig(filename_plot)
        plt.close(fig)
        end = time.time()

        wandb.log({"time": end - start})
        wandb.log({"expected_time_hours": ((end - start) * (frame_number - i))/3600})
        break
    break
