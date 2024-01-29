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

def calc_IoU(ground_truth, prediction):
    unique_classes = torch.unique(torch.concat((ground_truth, prediction)))
    iou_scores = []
    # Calculate IoU for each class
    for class_id in unique_classes:
        # binary mask for current class
        ground_truth_mask = (ground_truth == class_id)
        predicted_mask = (prediction == class_id)

        # intersection and union
        intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
        union = np.logical_or(predicted_mask, ground_truth_mask).sum()

        if union != 0:
            iou = intersection / union
            iou_scores.append(iou)
    return iou_scores


parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    help="Path to the folder with saved model",
    default="/home/guests/dominika_darabos/segmentation-for-cataract-surgery/trained_models/oneformer_ade20k_swin_tiny_20240109_175317"
)
args = parser.parse_args()

# Load the processor
processor_save_path = os.path.join(args.path, "processor")
processor = AutoProcessor.from_pretrained(processor_save_path)

# Load the model
model_save_path = os.path.join(args.path, "model")
model = AutoModelForUniversalSegmentation.from_pretrained(model_save_path, is_training=False)

model.eval()

test_dataset = CustomDataset(processor, video_numbers=[2])
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

data_iter = iter(test_dataloader)

# Get the first batch
batch, original_image, ground_truth_labels = next(data_iter)
ground_truth_labels = ground_truth_labels.squeeze(0)

# Drop the keys that are not accepted by the model in eval mode
keys_to_drop = ['mask_labels', 'class_labels', 'text_inputs']
batch_ = {key: value for key, value in batch.items() if key not in keys_to_drop}


# forward pass (no need for gradients at inference time)
with torch.no_grad():
    outputs = model(**batch_)

# postprocessing
segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[[540, 960]])[0]

# Metrics
iou_scores = calc_IoU(ground_truth_labels, segmentation)
mean_iou = np.mean(iou_scores) if iou_scores else 0

print("IoU values: ", iou_scores)
print("mIoU values: ", mean_iou)

# draw the segmentatin
viridis = cm.get_cmap('viridis', torch.max(torch.max(segmentation), torch.max(ground_truth_labels)))
fig, axs = plt.subplots(1,3, figsize=(18,6))

# first plot
original_image_squeezed = original_image.squeeze(0)
axs[0].imshow(original_image_squeezed)
axs[0].set_title("Original Image")

# second plot
axs[1].imshow(ground_truth_labels)
axs[1].set_title("Ground Truth")

# third plot
axs[2].imshow(segmentation)
axs[2].set_title("Predicted Segmentation")

# get all the unique numbers and labels
unique_patches = {}
labels_ids = torch.unique(segmentation).tolist()
ground_truth_labels_ids = torch.unique(ground_truth_labels).tolist()
for label_id in list(set(ground_truth_labels_ids + labels_ids)):
    label = processor.image_processor.metadata[str(label_id)]
    color = viridis(label_id)
    if label not in unique_patches:
        unique_patches[label] = mpatches.Patch(color=color, label=label)

handles = list(unique_patches.values())
fig.legend(handles=handles, loc='upper center', ncol=len(handles)//2, bbox_to_anchor=(0.5, 0.95))
plt.tight_layout()
plt.savefig(f'samples/sample_segmentation_{args.path.split("/")[-1]}.png')
