#!/bin/bash
#SBATCH --job-name=clip_food_exp3
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/vvjumle/logs/exp3_%j.out
#SBATCH --error=/scratch/vvjumle/logs/exp3_%j.err

set -e

REPO=/scratch/vvjumle/clip-food-good

module load 2025
module load miniconda3
conda activate ir-pipeline

export TORCH_HOME=/scratch/vvjumle/models
export RECIPE1M_IMAGE_ROOT=/scratch/vvjumle/clip-food-good/data/recipe1m
export RECIPE1M_NUM_WORKERS=4

cd $REPO
mkdir -p outputs

echo "=== Experiment 3: Lightweight Fine-tuning ==="
python -m src.run_experiment3
