#!/bin/bash
#SBATCH --job-name=cosmosis_gpu_boostf
#SBATCH --output=cosmosis_gpu_boost_%j.out
#SBATCH --error=cosmosis_gpu_boost_%j.err
#SBATCH --qos=regular                # or 'debug' for short runs
#SBATCH --nodes=1
#SBATCH --gpus=1                     # Request 1 GPU
#SBATCH --cpus-per-task=4           # Match your emcee config (10 walkers, 4 CPUs)
#SBATCH --time=01:00:00             # Max runtime (adjust as needed)
#SBATCH --constraint=gpu


# Load environment
module restore cosmosis
module load PrgEnv-gnu
module load cudatoolkit 


# Activate your virtual environment if needed
# source /path/to/venv/bin/activate

# Move to working directory
cd /its/home/aa3044/CosmoSIS_2025
source cosmosis_env/bin/activate
source cosmosis-configure

# (Optional) Confirm GPU is visible
nvcc --version
nvidia-smi

# Run the pipeline
cosmosis /its/home/aa3044/CUDA_course/CUDA_Programming/project/boost_factor.ini
