from transformers import AutoProcessor, AutoModelForUniversalSegmentation
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
import argparse
import wandb
import json

from custom_dataset import CustomDataset
from utils import EarlyStopping, prepare_metadata
from metrics import evaluate, test


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=int, help="Experiment number [1,2,3]", default=None)
parser.add_argument(
    "--model_name",
    help="Name of the model from HuggingFace model hub",
    default="oneformer_ade20k_swin_tiny"
)
parser.add_argument("--lr", type=float, help="Learning rate", default=5e-5)
parser.add_argument("--epochs", type=int, help="Number of epochs", default=1)
parser.add_argument("--batch_size", type=int, help="Batch size", default=1)
parser.add_argument("--mixed_precision", type=bool, help="Mixed precision", default=False)
args = parser.parse_args()

# Create save path
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"/home/data/cadis_results/trained_models/{args.model_name}_{timestamp}"

training_params = {
  "experiment": args.experiment,
  "model_name": args.model_name,
  "lr": args.lr,
  "epochs": args.epochs,
  "batch_size": args.batch_size,
  "mixed_precision": args.mixed_precision,
  "model_save_path": save_dir,
}
EVAL_STEPS = 100

wandb.login()

run = wandb.init(
    # Set the project where this run will be logged
    project="segmentation-for-cataract-surgery",
    # Track hyperparameters and run metadata
    config=training_params
)

processor = AutoProcessor.from_pretrained(f"shi-labs/{training_params['model_name']}")
# replace class mapping based on the experiment
with open(f"data/class_info/class_info_experiment{training_params['experiment']}.json", "r") as f:
    class_info = json.load(f)
processor.metadata = prepare_metadata(class_info)

model = AutoModelForUniversalSegmentation.from_pretrained(f"shi-labs/{training_params['model_name']}", is_training=True) 

processor.image_processor.num_text = model.config.num_queries - model.config.text_encoder_n_ctx

train_dataset = CustomDataset(processor, video_numbers=[1], experiment=training_params['experiment']) #3,4,5,8,9,10,11,13,14,15,17,18,19,20,21,23,24,25
val_dataset = CustomDataset(processor, video_numbers=[5], experiment=training_params['experiment']) #,7,16])
test_dataset = CustomDataset(processor, video_numbers=[2], experiment=training_params['experiment'])

train_dataloader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

optimizer = AdamW(model.parameters(), lr=training_params["lr"])
early_stopping = EarlyStopping(patience=1000, verbose=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
scaler = GradScaler()

device = "cuda" if torch.cuda.is_available() else "cpu"

model.train()
model.to(device)

for epoch in range(training_params["epochs"]):
    for step, (batch, _, _) in enumerate(train_dataloader):

        optimizer.zero_grad()

        batch = {k:v.to(device) for k,v in batch.items()}

        # forward pass
        if args.mixed_precision:
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
                # Scale the loss to prevent underflow
                scaler.scale(loss).backward()
                # Unscales the gradients and calls optimizer.step()
                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
        else:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        wandb.log({"loss": loss})

        if step % EVAL_STEPS == 0:  # Evaluate validation loss every eval_steps
            model.eval()
            avg_val_loss, mIoU, panoptic_quality, pac = evaluate(model, processor, val_dataloader, device)
            wandb.log({"val_loss": avg_val_loss, "mIoU": mIoU, "panoptic_quality": panoptic_quality, "pixel accuracy per class": pac})
            model.train()
            if early_stopping.should_stop(avg_val_loss):
                break

    scheduler.step(avg_val_loss)

# Save the trained model
model_save_path = f"{save_dir}/model"
model.save_pretrained(model_save_path)

# Save the processor (if needed for inference)
processor_save_path = f"{save_dir}/processor"
processor.save_pretrained(processor_save_path)

# Save other relevant information
torch.save(optimizer.state_dict(), f"{save_dir}/optimizer.pth")

# Log the saved model and other files to W&B
artifact = wandb.Artifact(
    name=f"{training_params['model_name']}_{timestamp}",
    type="model",
    description=f"Trained model for {training_params['model_name']} at epoch {epoch + 1}",
)
artifact.add_dir(model_save_path)
artifact.add_file(f"{save_dir}/optimizer.pth")
wandb.log_artifact(artifact)

mIoU, panoptic_quality, pac = test(model, processor, test_dataloader, device, drop_text_model=False)
wandb.log({"test_mIoU": mIoU, "test_panoptic_quality": panoptic_quality, "test_pixel accuracy per class": pac})
