#!/bin/bash
#SBATCH --job-name=clip_food_exp2
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/vvjumle/logs/exp2_%j.out
#SBATCH --error=/scratch/vvjumle/logs/exp2_%j.err

set -e

REPO=/scratch/vvjumle/clip-food-good

module load 2025
module load miniconda3
conda activate CV

export TORCH_HOME=/scratch/vvjumle/models
export RECIPE1M_IMAGE_ROOT=/scratch/vvjumle/clip-food-good/data/recipe1m
export RECIPE1M_NUM_WORKERS=4

cd $REPO
mkdir -p outputs

echo "=== Experiment 2: Prompt Engineering ==="
python -m src.run_experiment2
