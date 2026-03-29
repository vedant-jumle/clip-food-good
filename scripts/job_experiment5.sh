#!/bin/bash
#SBATCH --job-name=clip_food_exp5
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=2
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/vvjumle/logs/exp5_%j.out
#SBATCH --error=/scratch/vvjumle/logs/exp5_%j.err

set -e

REPO=/scratch/vvjumle/clip-food-good

module load 2025
module load miniconda3
module load cuda
conda activate CV

export TORCH_HOME=/scratch/vvjumle/models
export RECIPE1M_IMAGE_ROOT=/scratch/vvjumle/clip-food-good/data/recipe1m
export RECIPE1M_LAYER2=/scratch/vvjumle/clip-food-good/data/recipe1m/layer2+.json
export RECIPE1M_NUM_WORKERS=8
export RECIPE1M_MAX_TRAIN_SAMPLES=150000
export RECIPE1M_EPOCHS=80

cd $REPO
mkdir -p outputs

echo "=== Experiment 5: Asymmetric Frequency-Weighted Fine-tuning ==="
python -m src.run_experiment5
