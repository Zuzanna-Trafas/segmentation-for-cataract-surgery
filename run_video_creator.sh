#!/bin/sh
 
#SBATCH --job-name=video_creator
#SBATCH --output=video_creator.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=video_creator.err  # Standard error of the script
#SBATCH --time=7-12:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:RTX3090:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=2  # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=32G  # Memory in GB (Don't use more than 48/24 per GPU unless you absolutely need it and know what you are doing)
 
# To run a script on the cluster, change CONDA_PATH and export WANDB_API_KEY
ml cuda  # load default CUDA module
CONDA_PATH=/home/guests/dominika_darabos/miniconda3
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate oneformer
python src/video_creator.py --input_folder="/home/data/cadis_results/segmented_images/Comparison" --output_path="/home/guests/dominika_darabos/segmentation-for-cataract-surgery/samples/" --frame_rate=10 --every_n=2 --video_name="segmented_size_comparison_01" --start=0 --end=4000
ml -cuda  # unload all modules
conda deactivate