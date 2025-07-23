#!/bin/bash
#SBATCH --job-name=cosmosis_gpu_boost
#SBATCH --output=cosmosis_gpu_boost_%j.out
#SBATCH --error=cosmosis_gpu_boost_%j.err
#SBATCH --qos=regular                # or 'debug' for short runs
#SBATCH --nodes=1
#SBATCH --gpus=1                     # Request 1 GPU
#SBATCH --cpus-per-task=4           # Match your emcee config (10 walkers, 4 CPUs)
#SBATCH --time=02:00:00             # Max runtime (adjust as needed)
#SBATCH --constraint=gpu
#SBATCH --account=your_project_name_here

# Load environment
module load cuda
module load python             # Or your preferred module set

# Activate your virtual environment if needed
# source /path/to/venv/bin/activate

# Move to working directory
cd /global/homes/a/arwa_mq/DESy3/Boost_factor_cosmosis

# (Optional) Confirm GPU is visible
nvidia-smi

# Run the pipeline
cosmosis boost_factor.ini
