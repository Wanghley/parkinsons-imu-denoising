#!/bin/bash
#SBATCH --job-name=ece685_full_run
#SBATCH --output=logs/run_%j.out
#SBATCH --error=logs/run_%j.err
#SBATCH -p gpu-common
#SBATCH --account=agashelab
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# 1. Prevent 'ctypes' or local library conflicts on the cluster
export PYTHONNOUSERSITE=1

# 2. Create a logs directory if it doesn't exist
mkdir -p logs

# 3. Activate the virtual environment
# Note: Using the absolute path provided to ensure it works from any directory
source /hpc/home/ws186/project-685/.venv/bin/activate

# 4. Generate a dynamic tag based on hardware and start time
# This detects the GPU (e.g., GeForce_RTX_2080_Ti) and appends a short timestamp
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr ' ' '_' | sed 's/[^a-zA-Z0-9_]//g' | head -n 1)
RUN_TIME=$(date +%H%M)
DYNAMIC_TAG="${GPU_NAME}_${RUN_TIME}"

# 5. Run the full experiment suite
python main.py --exp all --tag "$DYNAMIC_TAG"
