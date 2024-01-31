import torch
import numpy as np


def calculate_mIoU(predicted, target, num_classes=36):
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


def calculate_pixel_accuracy_per_class(predicted, target, num_classes=36):
    class_pixel_accuracy = torch.zeros(num_classes)
    for cls in range(num_classes):
        correct_pixels = torch.sum((predicted == cls) & (target == cls))
        total_pixels = torch.sum(target == cls)
        if total_pixels != 0:
            class_pixel_accuracy[cls] = correct_pixels / total_pixels
    return class_pixel_accuracy.mean().item()


def evaluate(model, processor, val_dataloader, device, num_classes=36, void_label=255):
    total_val_loss = 0.0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for i, (val_batch, _, target) in enumerate(val_dataloader):
            val_batch = {k: v.to(device) for k, v in val_batch.items()}

            outputs = model(**val_batch)
            val_loss = outputs.loss
            total_val_loss += val_loss.item()

            # TODO panoptic?
            predictions = processor.post_process_semantic_segmentation(outputs, target_sizes=[[540, 960]])

            all_predictions.append(predictions[0].cpu())
            all_targets.append(target[0].cpu())

    avg_val_loss = total_val_loss / len(val_dataloader)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    mIoU = calculate_mIoU(all_predictions, all_targets, num_classes)
    panoptic_quality = calculate_panoptic_quality(all_predictions, all_targets, void_label)
    pac = calculate_pixel_accuracy_per_class(all_predictions, all_targets, num_classes)

    return avg_val_loss, mIoU, panoptic_quality, pac


def test(model, processor, test_dataloader, device, drop_text_model=True, num_classes=36, void_label=255):
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for i, (val_batch, _, target) in enumerate(test_dataloader):
            if drop_text_model:
                keys_to_drop = ['mask_labels', 'class_labels', 'text_inputs']
                val_batch = {key: value for key, value in val_batch.items() if key not in keys_to_drop}
            
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            outputs = model(**val_batch)

            # TODO panoptic?
            predictions = processor.post_process_semantic_segmentation(outputs, target_sizes=[[540, 960]])

            all_predictions.append(predictions[0].cpu())
            all_targets.append(target[0].cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    mIoU = calculate_mIoU(all_predictions, all_targets, num_classes)
    panoptic_quality = calculate_panoptic_quality(all_predictions, all_targets, void_label)
    pac = calculate_pixel_accuracy_per_class(all_predictions, all_targets, num_classes)

    return mIoU, panoptic_quality, pac
