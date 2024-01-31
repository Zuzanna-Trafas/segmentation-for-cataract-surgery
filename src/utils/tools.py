import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import torch
import json


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
    viridis = cm.get_cmap(
        "viridis", torch.max(torch.max(segmentation), torch.max(ground_truth_labels))
    )
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

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
        color = viridis(label_id)
        if label not in unique_patches:
            unique_patches[label] = mpatches.Patch(color=color, label=label)

    handles = list(unique_patches.values())
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles) // 2,
        bbox_to_anchor=(0.5, 0.95),
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