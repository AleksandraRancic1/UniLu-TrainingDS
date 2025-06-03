#!/bin/bash
#SBATCH --job-name=cnn_gpu
#SBATCH --output=logs/cnn_gpu_%j.out
#SBATCH --error=logs/cnn_gpu_%j.err
#SBATCH --partition=gpu
#SBATCH --reservation=school-d2-gpu
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --cpus-per-task=4          # Use 4 CPUs for data loading
#SBATCH --mem=16G                  # Memory for the job
#SBATCH --time=01:00:00            # Max time (adjust as needed)

module --force purge

module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0
module load data/scikit-learn/1.4.0-gfbf-2023b
module load lang/SciPy-bundle/2023.11-gfbf-2023b
module load vis/matplotlib/3.8.2-gfbf-2023b
module load bio/Seaborn/0.13.2-gfbf-2023b

# Run your GPU-based training script
python cnn_gpu.py
