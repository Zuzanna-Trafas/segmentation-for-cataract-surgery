import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import torch
import json
from utils.cadis_visualization import get_cadis_colormap
import numpy as np
from matplotlib.colors import ListedColormap

def apply_custom_colormap(binary, colormap):
    # segmentation is 2D (H, W) instead of 3D (C, H, W)
    if binary.ndim == 2:
        binary = binary.unsqueeze(0)  # Add C dim

    # base image with the same width and height as our segmentation mask
    height, width = binary.shape[1], binary.shape[2]
    colored_segmentation = np.zeros((height, width, 3), dtype=np.uint8)

    for label_index, color in enumerate(colormap):
        mask = binary[0] == label_index
        colored_segmentation[mask] = color

    return colored_segmentation


def save_segmentation(original_image, ground_truth_labels, segmentation, miou, processor, path):
    """
    Save visualization of segmentation results along with original image and ground truth labels.

    Args:
        original_image (torch.Tensor): Original image tensor.
        ground_truth_labels (torch.Tensor): Ground truth segmentation labels tensor.
        segmentation (torch.Tensor): Predicted segmentation tensor.
        miou (float): Mean Intersection over Union (mIoU) value.
        processor: The processor for post-processing segmentation results.
        path (str): Path to save the visualization.

    """
    colormap = get_cadis_colormap()
    custom_cmap = ListedColormap(colormap / 255.5)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # first plot
    original_image_squeezed = original_image.squeeze(0)
    axs[0].imshow(original_image_squeezed)
    axs[0].set_title("Original Image")

    # second plot
    colored_ground_truth_labels = apply_custom_colormap(ground_truth_labels, colormap)
    axs[1].imshow(colored_ground_truth_labels)
    axs[1].set_title("Ground Truth")

    # third plot
    colored_segmentation = apply_custom_colormap(segmentation, colormap)
    axs[2].imshow(colored_segmentation)
    axs[2].set_title("Predicted Segmentation")
    # Annotate with mIoU value
    axs[2].text(
        0.5, -0.2, f"mIoU: {miou}", fontsize=12, ha="center", transform=axs[2].transAxes
    )

    # get all the unique numbers and labels
    unique_patches = {}
    labels_ids = torch.unique(segmentation).tolist()
    ground_truth_labels_ids = torch.unique(ground_truth_labels).tolist()
    for label_id in list(set(ground_truth_labels_ids + labels_ids)):
        label = processor.image_processor.metadata[str(label_id)]
        color = custom_cmap(label_id)
        if label not in unique_patches:
            unique_patches[label] = mpatches.Patch(color=color, label=label)

    handles = list(unique_patches.values())
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles) // 2,
        bbox_to_anchor=(0.5, 0.95),
        edgecolor='black',
    )
    plt.tight_layout()
    plt.savefig(path)

def prepare_metadata(experiment):
    """
    Prepare metadata that defines the processor class names and ids for the given experiment

    Args:
        experiment (int): Experiment number from [1,2,3]

    Returns:
        dict: Metadata dictionary containing class information.
    """
    with open(f"data/class_info/class_info_experiment{experiment}.json", "r") as f:
        class_info = json.load(f)
    metadata = {}
    class_names = []
    thing_ids = []
    for key, info in class_info.items():
        metadata[key] = info["name"]
        class_names.append(info["name"])
        if info["isthing"]:
            thing_ids.append(int(key))
    metadata["thing_ids"] = thing_ids
    metadata["class_names"] = class_names
    return metadata


def save_checkpoint(model, processor, optimizer, path):
    """
    Save model to the given path.

    Args:
        model: Model to save.
        processor: Processor to save.
        optimizer: Optimizer to save.
        path (str): Path to save the model.
    """
    # Save the trained model
    model_save_path = f"{path}/model"
    model.save_pretrained(model_save_path)

    # Save the processor
    processor_save_path = f"{path}/processor"
    processor.save_pretrained(processor_save_path)

    # Save other relevant information
    torch.save(optimizer.state_dict(), f"{path}/optimizer.pth")