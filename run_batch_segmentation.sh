#!/bin/sh
 
#SBATCH --job-name=inference-onformer
#SBATCH --output=inference-onformer.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=inference-onformer.err  # Standard error of the script
#SBATCH --time=2-12:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:RTX3090:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=2  # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=32G  # Memory in GB (Don't use more than 48/24 per GPU unless you absolutely need it and know what you are doing)
 
# To run a script on the cluster, change CONDA_PATH and export WANDB_API_KEY
ml cuda  # load default CUDA module
CONDA_PATH=/home/guests/zuzanna_trafas/anaconda3
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate oneformer
python src/batch_segmentation.py --model_folder_name oneformer_ade20k_swin_large_20240305_151306 --output_dir /home/data/cadis_results/segmentation_results --video_id 3
ml -cuda  # unload all modules
conda deactivate