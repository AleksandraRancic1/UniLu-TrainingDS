#!/bin/bash
#SBATCH --job-name=ddp_cnn
#SBATCH --output=logs/ddp_cnn_%j.out
#SBATCH --error=logs/ddp_cnn_%j.err
#SBATCH --partition=gpu
#SBATCH --reservation=school-d2-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

module --force purge

module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0
module load data/scikit-learn/1.4.0-gfbf-2023b
module load lang/SciPy-bundle/2023.11-gfbf-2023b
module load vis/matplotlib/3.8.2-gfbf-2023b
module load bio/Seaborn/0.13.2-gfbf-2023b

# Use torchrun
torchrun \
    --nnodes=1 \
    --nproc-per-node=2 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:29500 \
    cnn_ddp_gpu.py
