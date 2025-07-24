#!/bin/bash
#SBATCH --job-name=bf_job
#SBATCH --output=bf_m_output.log
#SBATCH --error=job_error.log
#SBATCH --time=01:00:00            # 1 hour
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # Total tasks (processes)
#SBATCH --cpus-per-task=4         # Number of threads per task
#SBATCH --mem=4G                  # Memory per node

# Load modules if needed
# module restore cosmosis


# Activate conda environment (optional)
cd /its/home/aa3044/CosmoSIS_2025
source cosmosis_env/bin/activate
source cosmosis-configure

# Run your command
cd /its/home/aa3044/CUDA_course/CUDA_Programming/project

cosmosis boost_factor.ini