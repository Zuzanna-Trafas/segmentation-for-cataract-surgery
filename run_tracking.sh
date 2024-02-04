#!/bin/sh

#SBATCH --job-name=tracking-run
#SBATCH --output=tracking.out         # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=tracking.err          # Standard error of the script
#SBATCH --time=0-12:00:00             # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1                  # Number of GPUs if needed
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks (processes) per node
#SBATCH --cpus-per-task=2             # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=32G                     # Memory in GB (Don't use more than 48/24 per GPU unless you absolutely need it and know what you are doing)


# run the program
ml cuda  # load default CUDA module
CONDA_PATH=/home/guests/nguyentoan_le/anaconda3
source $CONDA_PATH/etc/profile.d/conda.sh
# ml miniconda3  # load default miniconda and python module
if command -v conda &> /dev/null; then
    conda activate OneFormerVirEnvSSH38
else
    source activate OneFormerVirEnvSSH38
fi
python src/tracking.py --object_id=1 --name_trajectory "Video_2_Tool_1.png" --segmentation "/home/data/CaDISv2/Video02/Labels" --output "/home/guests/nguyentoan_le/Praktikum"
ml -cuda  # unload all modules
conda deactivate
