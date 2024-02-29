#!/bin/sh
 
#SBATCH --job-name=pupil_size_calculator
#SBATCH --output=pupil_size_calculator.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=pupil_size_calculator.err  # Standard error of the script
#SBATCH --time=7-12:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:RTX3090:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=2  # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=32G  # Memory in GB (Don't use more than 48/24 per GPU unless you absolutely need it and know what you are doing)

ml cuda  # load default CUDA module
CONDA_PATH=/home/guests/dominika_darabos/miniconda3
source $CONDA_PATH/etc/profile.d/conda.sh
# ml miniconda3  # load default miniconda and python module
if command -v conda &> /dev/null; then
    conda activate oneformer
else
    source activate oneformer
fi
python3 src/pupil_size_calculator.py --input_folder="/home/data/CaDISv2/Video01/Labels" --output_folder="/home/guests/dominika_darabos/segmentation-for-cataract-surgery/samples/pupil_size" --change_threshold=10