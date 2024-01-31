# Panoptic segmentation for Catarct surgery

Project as part of Computational Surgineering at TUM. The project goals are:

- Perform panoptic segmentation on the "*CaDIS: Cataract dataset for surgical RGB-image segmentation*" dataset
- Calcuate the size of the pupil to provide information on sudden changes
- Tracking of instruments to perform Economy of Motion statistics calculation

## How to run
To fine-tune the model on the IFL cluster (or using any SLURM cluster), redefine CONDA_PATH, environment, and WANDB_API_KEY in `run.sh`. Specify training parameters from the following:

- --experiment: Experiment number [1,2,3] (default: None, no task remapping is applied)
- --model_name: Name of the model from HuggingFace model hub (default: "oneformer_ade20k_swin_tiny")
- --lr: Learning rate (default: 5e-5)
- --epochs: Number of epochs (default: 1)
- --batch_size: Batch size (default: 1)
- --mixed_precision: Whether mixed precision is applied for training (default: False)

And run `sbatch run.sh`