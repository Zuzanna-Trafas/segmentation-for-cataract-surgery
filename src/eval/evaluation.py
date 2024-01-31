import torch
from eval.metrics import calculate_mIoU, calculate_panoptic_quality, calculate_pixel_accuracy_per_class


def evaluate(model, processor, val_dataloader, device):
    """
    Evaluate the given model on the validation dataset.

    Args:
        model: The model to evaluate.
        processor: The processor for post-processing semantic segmentation outputs.
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        device (torch.device): The device to run the evaluation on.

    Returns:
        tuple: A tuple containing:
            - avg_val_loss (float): The average validation loss.
            - mIoU (float): The mean Intersection over Union (mIoU) score.
            - panoptic_quality (float): The panoptic quality score.
            - pac (float): The pixel accuracy per class.

    """
    total_val_loss = 0.0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for (val_batch, _, target) in val_dataloader:
            val_batch = {k: v.to(device) for k, v in val_batch.items()}

            outputs = model(**val_batch)
            val_loss = outputs.loss
            total_val_loss += val_loss.item()

            predictions = processor.post_process_semantic_segmentation(outputs, target_sizes=[[540, 960]])

            all_predictions.append(predictions[0].cpu())
            all_targets.append(target[0].cpu())

    avg_val_loss = total_val_loss / len(val_dataloader)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    mIoU = calculate_mIoU(all_predictions, all_targets)
    panoptic_quality = calculate_panoptic_quality(all_predictions, all_targets)
    pac = calculate_pixel_accuracy_per_class(all_predictions, all_targets)

    return avg_val_loss, mIoU, panoptic_quality, pac


def test(model, processor, test_dataloader, device):
    """
    Test the given model on the test dataset.

    Args:
        model: The model to evaluate (after loading model weights in non-training mode)
        processor: The processor for post-processing semantic segmentation outputs.
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        device (torch.device): The device to run the evaluation on.

    Returns:
        tuple: A tuple containing:
            - mIoU (float): The mean Intersection over Union (mIoU) score.
            - panoptic_quality (float): The panoptic quality score.
            - pac (float): The pixel accuracy per class.

    """
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for (val_batch, _, target) in test_dataloader:
            keys_to_drop = ['mask_labels', 'class_labels', 'text_inputs']
            val_batch = {key: value for key, value in val_batch.items() if key not in keys_to_drop}
            
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            outputs = model(**val_batch)

            predictions = processor.post_process_semantic_segmentation(outputs, target_sizes=[[540, 960]])

            all_predictions.append(predictions[0].cpu())
            all_targets.append(target[0].cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    mIoU = calculate_mIoU(all_predictions, all_targets)
    panoptic_quality = calculate_panoptic_quality(all_predictions, all_targets)
    pac = calculate_pixel_accuracy_per_class(all_predictions, all_targets)

    return mIoU, panoptic_quality, pac
