#!/usr/bin/env bash
#SBATCH -J fusion
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:3
#SBATCH -c 8
#SBATCH -t 0-3:00:00
#SBATCH --mem=80G
#SBATCH -o logs/forces_nn_%j.out
#SBATCH -e logs/forces_nn_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aurora_ruggeri@seas.harvard.edu

# Create logs and outputs directories
mkdir -p logs
mkdir -p outputs

# Load modules
module load python/3.10.12-fasrc01

# Activate virtual environment
source ~/tactile_sensor/venv/bin/activate

# Prevent BLAS from stealing threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Print GPU info
echo "=== GPU Info ==="
nvidia-smi
echo "================"

# Run training script
python code/forces_image_baro_fusion.py