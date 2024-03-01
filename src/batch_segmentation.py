from transformers import AutoProcessor, AutoModelForUniversalSegmentation
from torch.utils.data import DataLoader
from cadis_dataset import CadisDataset, CataractsDataset

import argparse
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import time
from utils.tools import apply_custom_colormap
from utils.cadis_visualization import get_cadis_colormap

#start = time.time()
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
#stop_1 = time.time()
#print(stop_1 - start)
colormap = get_cadis_colormap()

for video_num in [2, 12, 22]:
    test_dataset = CataractsDataset(processor, video_numbers=[video_num])
    stop_2 = time.time()
    #print(stop_2 - stop_1)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    stop_3 = time.time()
    #print(stop_3 - stop_2)
    for i, (batch, original_image, frame_image_path) in enumerate(test_dataloader):
        batch_ = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch_)

        # Postprocessing
        segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[[540, 960]])[0]
        frame = frame_image_path.split("/")[0][-1]
        frame = frame.split(".")[0]
        filename = os.path.join(args.output_dir, f'Video_{video_num:02d}', f'seg_{video_num}_{frame}.png')
        #segmentation_image = Image.fromarray(segmentation.numpy())
        #segmentation_image.save(filename)
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        segmentation_squeezed = segmentation.squeeze(0)
        segmentation_squeezed = apply_custom_colormap(segmentation_squeezed, colormap)
        axs.imshow(segmentation_squeezed.cpu().detach())
        axs.axis("off")
        plt.tight_layout()
        plt.savefig(filename)

        filename_compare = os.path.join(args.output_dir, f'Comparison/Video_{video_num:02d}',
                                        f'seg_{video_num}_{frame}.png')
        fig2, axs2 = plt.subplots(1, 2, figsize=(12, 6))
        # Original plot
        original_image_squeezed = original_image.squeeze(0)
        axs2[0].imshow(original_image_squeezed)
        axs2[0].set_title("Original Image")
        # Segmentation plot
        axs2[1].imshow(segmentation_squeezed.cpu().detach())
        axs2[1].set_title("Segmented Image")
        axs2.axis("off")
        plt.tight_layout()
        plt.savefig(filename_compare)


print("Finished!!!")
