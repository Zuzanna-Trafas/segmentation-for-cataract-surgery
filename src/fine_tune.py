from transformers import AutoProcessor, AutoModelForUniversalSegmentation
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datetime import datetime
import argparse
import wandb

from custom_dataset import CustomDataset


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    help="Name of the model from HuggingFace model hub",
    default="oneformer_ade20k_swin_tiny"#"oneformer_coco_swin_large"
)
parser.add_argument("--lr", type=float, help="Learning rate", default=5e-5)
parser.add_argument("--epochs", type=int, help="Number of epochs", default=1)
parser.add_argument("--batch_size", type=int, help="Batch size", default=1)
args = parser.parse_args()

training_params = {
  "model_name": args.model_name,
  "lr": args.lr,
  "epochs": args.epochs,
  "batch_size": args.batch_size,
}

wandb.login()

run = wandb.init(
    # Set the project where this run will be logged
    project="segmentation-for-cataract-surgery",
    # Track hyperparameters and run metadata
    config=training_params
)

processor = AutoProcessor.from_pretrained(f"shi-labs/{training_params['model_name']}")
"""
 TODO the previous line might automatically cache
 /home/guests/<user>/.cache/huggingface/hub/datasets--shi-labs--oneformer_demo/snapshots/4d683bd5bf84e9c8b5537dce306230bde409fe89/coco_panoptic.json
the contents of the file need to be replaced with contents of data/class_info.json if that happens
"""
processor.image_processor.class_info_file = 'class_info.json'
processor.image_processor.repo_path = 'ztrafas/segmentation-for-cataract-surgery'

model = AutoModelForUniversalSegmentation.from_pretrained(f"shi-labs/{training_params['model_name']}", is_training=True) 

processor.image_processor.num_text = model.config.num_queries - model.config.text_encoder_n_ctx

train_dataset = CustomDataset(processor, video_numbers=[1]) #[1,3,4,5,8,9,10,11,13,14,15,17,18,19,20,21,23,24,25]]
val_dataset = CustomDataset(processor, video_numbers=[5]) #[5,7,16]
test_dataset = CustomDataset(processor, video_numbers=[2]) #[2,12,22]

train_dataloader = DataLoader(train_dataset, batch_size=training_params["batch_size"], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=training_params["batch_size"], shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=training_params["batch_size"], shuffle=False)

optimizer = AdamW(model.parameters(), lr=training_params["lr"])

device = "cuda" if torch.cuda.is_available() else "cpu"

print("STARTING TRAINING")
model.train()
model.to(device)

for epoch in range(training_params["epochs"]):  # loop over the dataset multiple times
    for batch in train_dataloader:

        # zero the parameter gradients
        optimizer.zero_grad()

        batch = {k:v.to(device) for k,v in batch.items()}

        # forward pass
        outputs = model(**batch)

        # backward pass + optimize
        loss = outputs.loss
        # TODO log other metrics
        wandb.log({"loss": loss})
        loss.backward()
        optimizer.step()
      
    # Validation loop
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for val_batch in val_dataloader:
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            val_outputs = model(**val_batch)
            val_loss = val_outputs.loss
            total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(val_dataloader)
    wandb.log({"val_loss": avg_val_loss})

# Save the trained model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"trained_models/{training_params['model_name']}_{timestamp}"
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
