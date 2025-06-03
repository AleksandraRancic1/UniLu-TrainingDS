#!/bin/bash
#SBATCH --job-name=cnn_cpu_parallel
#SBATCH --output=logs/cnn_cpu_parallel_%j.out
#SBATCH --error=logs/cnn_cpu_parallel_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --reservation=school-d2-batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G


module --force purge

module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b
module load data/scikit-learn/1.4.0-gfbf-2023b
module load lang/SciPy-bundle/2023.11-gfbf-2023b
module load vis/matplotlib/3.8.2-gfbf-2023b
module load bio/Seaborn/0.13.2-gfbf-2023b

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run
python cnn_cpu_parallel.py
