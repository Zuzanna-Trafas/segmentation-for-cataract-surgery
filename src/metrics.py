import torch
from sklearn.metrics import average_precision_score
import numpy as np


def calculate_mIoU(predicted, target, num_classes):
    ious = []
    for cls in range(num_classes):
        intersection = ((predicted == cls) & (target == cls)).sum().item()
        union = ((predicted == cls) | (target == cls)).sum().item()
        if union != 0:
            iou = intersection / union
            ious.append(iou)
    if len(ious) == 0:
        return 0.0
    else:
        return np.mean(ious)


def calculate_panoptic_quality(predicted, target, void_label):
    intersection = ((predicted != void_label) & (target != void_label) & (predicted == target)).sum().item()
    union = ((predicted != void_label) | (target != void_label)).sum().item()
    if union != 0:
        pq = intersection / union
        return pq
    else:
        return 0.0


# def calculate_average_precision(predictions, targets, num_classes):
#     all_probabilities = torch.sigmoid(predictions)
#     all_probabilities = all_probabilities[:, 1:]  # Remove background class if present

#     targets = targets[:, 1:]  # Remove background class if present
#     targets = targets.view(-1, num_classes - 1)

#     all_probabilities = all_probabilities.view(-1, num_classes - 1)

#     average_precision = average_precision_score(targets.numpy(), all_probabilities.numpy(), average='micro')
#     return average_precision


def calculate_pixel_accuracy_per_class(predicted, target, num_classes):
    class_pixel_accuracy = torch.zeros(num_classes)
    for cls in range(num_classes):
        correct_pixels = torch.sum((predicted == cls) & (target == cls))
        total_pixels = torch.sum(target == cls)
        if total_pixels != 0:
            class_pixel_accuracy[cls] = correct_pixels / total_pixels
    return class_pixel_accuracy.mean().item()


# def concat_class_channels(mask_labels, class_labels):
#     result = torch.zeros((mask_labels.shape[-2], mask_labels.shape[-1]), dtype=torch.int64)
#     for label, mask in zip(class_labels, mask_labels):
#         result[mask == 1] = label

#     return result


def evaluate(model, processor, val_dataloader, device, num_classes=36, void_label=255):
    model.eval()
    total_val_loss = 0.0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for val_batch in val_dataloader:
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            outputs = model(**val_batch)
            val_loss = outputs.loss
            total_val_loss += val_loss.item()

            # Assuming the model returns 'predictions' and 'targets'
            predictions = processor.post_process_semantic_segmentation(outputs, target_sizes=[[540, 960]])[0]
            target = val_batch['target'][0]

            all_predictions.append(predictions.cpu())
            all_targets.append(target.cpu())

    avg_val_loss = total_val_loss / len(val_dataloader)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    mIoU = calculate_mIoU(all_predictions, all_targets, num_classes)
    panoptic_quality = calculate_panoptic_quality(all_predictions, all_targets, void_label)
    pac = calculate_pixel_accuracy_per_class(all_predictions, all_targets, num_classes)
    # average_precision_class = calculate_average_precision(all_predictions, all_targets, num_classes)

    return avg_val_loss, mIoU, panoptic_quality, pac
