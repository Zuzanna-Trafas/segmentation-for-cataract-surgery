from transformers import AutoProcessor, AutoModelForUniversalSegmentation
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datetime import datetime
import torch
import argparse
import wandb
import os

from constants import MODEL_SAVE_DIR, EVAL_STEPS
from cadis_dataset import CadisDataset
from utils.early_stopping import EarlyStopping
from utils.tools import prepare_metadata, save_checkpoint
from eval.evaluation import evaluate


# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment", type=int, help="Experiment number [1,2,3]", default=None
)
parser.add_argument(
    "--model_name",
    help="Name of the model from HuggingFace model hub",
    default="oneformer_ade20k_swin_tiny",
)
parser.add_argument("--lr", type=float, help="Learning rate", default=5e-5)
parser.add_argument("--epochs", type=int, help="Number of epochs", default=1)
parser.add_argument("--batch_size", type=int, help="Batch size", default=1)
parser.add_argument(
    "--mixed_precision", type=bool, help="Mixed precision", default=False
)
args = parser.parse_args()

# Create save path for the results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"{MODEL_SAVE_DIR}/{args.model_name}_{timestamp}"

training_params = {
    "experiment": args.experiment,
    "model_name": args.model_name,
    "lr": args.lr,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "mixed_precision": args.mixed_precision,
    "model_save_path": save_dir,
}

# Initialize wand and save hyperparameters
wandb.login()

run = wandb.init(
    project="segmentation-for-cataract-surgery",
    config=training_params,
)

# Load the model and processor from the HuggingFace model hub
processor = AutoProcessor.from_pretrained(f"shi-labs/{training_params['model_name']}")
processor.metadata = prepare_metadata(training_params['experiment']) # replace class mapping based on the experiment

model = AutoModelForUniversalSegmentation.from_pretrained(
    f"shi-labs/{training_params['model_name']}", is_training=True
)

processor.image_processor.num_text = (
    model.config.num_queries - model.config.text_encoder_n_ctx
)

# Load the datasets
train_dataset = CadisDataset(
    processor, 
    video_numbers=[1,3,4,5,8,9,10,11,13,14,15,17,18,19,20,21,23,24,25],
    experiment=training_params["experiment"]
)
val_dataset = CadisDataset(
    processor, 
    video_numbers=[6,7,16], 
    experiment=training_params["experiment"]
)
test_dataset = CadisDataset(
    processor, 
    video_numbers=[2,12,22], 
    experiment=training_params["experiment"]
)

# Create dataloaders
train_dataloader = DataLoader(
    train_dataset, batch_size=training_params["batch_size"], shuffle=True
)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize optimizer, scheduler, and early stopping
optimizer = AdamW(model.parameters(), lr=training_params["lr"])
early_stopping = EarlyStopping(patience=1000)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2000)
scaler = GradScaler() # for mixed precision training

device = "cuda" if torch.cuda.is_available() else "cpu"

model.train()
model.to(device)

best_val_loss = float('inf')
for epoch in range(training_params["epochs"]):
    for step, (batch, _, _) in enumerate(train_dataloader):
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}

        if args.mixed_precision:
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        scheduler.step(loss) # decrease lr if conditions are met
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"loss": loss, "lr": current_lr})

        if step % EVAL_STEPS == 0:  # Evaluate validation loss every EVAL_STEPS
            model.eval()
            val_loss, mIoU, panoptic_quality, pac = evaluate(
                model, processor, val_dataloader, device
            )
            wandb.log(
                {
                    "val loss": val_loss,
                    "mIoU": mIoU,
                    "panoptic quality": panoptic_quality,
                    "pixel accuracy per class": pac,
                }
            )
            model.train()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the model checkpoint
                checkpoint_path = os.path.join(save_dir, f"model_checkpoint_{epoch + 1}_{step}")
                save_checkpoint(model, processor, optimizer, checkpoint_path)
            if early_stopping.should_stop(val_loss):
                break


save_checkpoint(model, processor, optimizer, save_dir)

# Test the model
test_loss, mIoU, panoptic_quality, pac = evaluate(
                model, processor, test_dataloader, device
            )
wandb.log(
    {
        "test mIoU": mIoU,
        "test panoptic_quality": panoptic_quality,
        "test pixel accuracy per class": pac,
    }
)
