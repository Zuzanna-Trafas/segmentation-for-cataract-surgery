import torch
import numpy as np


def calculate_mIoU(predicted, target):
    """
    Calculate the mean Intersection over Union (mIoU) metric.

    Args:
        predicted (torch.Tensor): Predicted segmentation map.
        target (torch.Tensor): Ground truth segmentation map.

    Returns:
        float: The mean Intersection over Union (mIoU) score.
    """
    ious = []
    classes = torch.unique(target)
    for cls in classes:
        intersection = ((predicted == cls) & (target == cls)).sum().item()
        union = ((predicted == cls) | (target == cls)).sum().item()
        if union != 0:
            iou = intersection / union
            ious.append(iou)
    if len(ious) == 0:
        return 0.0
    else:
        return np.mean(ious)


def calculate_panoptic_quality(predicted, target, void_label=-1):
    """
    Calculate the Panoptic Quality (PQ) metric for semantic segmentation.

    Args:
        predicted (torch.Tensor): Predicted segmentation map.
        target (torch.Tensor): Ground truth segmentation map.
        void_label (int, optional): Label for void/ignore regions. Defaults to -1.

    Returns:
        float: The Panoptic Quality (PQ) score.
    """
    intersection = ((predicted != void_label) & (target != void_label) & (predicted == target)).sum().item()
    union = ((predicted != void_label) | (target != void_label)).sum().item()
    if union != 0:
        pq = intersection / union
        return pq
    else:
        return 0.0


def calculate_pixel_accuracy_per_class(predicted, target):
    """
    Calculate the pixel accuracy per class metric for semantic segmentation.

    Args:
        predicted (torch.Tensor): Predicted segmentation map.
        target (torch.Tensor): Ground truth segmentation map.

    Returns:
        float: The pixel accuracy per class.
    """
    classes = torch.unique(target)
    class_pixel_accuracy = torch.zeros(len(classes))
    for i, cls in enumerate(classes):
        correct_pixels = torch.sum((predicted == cls) & (target == cls))
        total_pixels = torch.sum(target == cls)
        if total_pixels != 0:
            class_pixel_accuracy[i] = correct_pixels / total_pixels
    return class_pixel_accuracy.mean().item()
