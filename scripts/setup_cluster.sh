#!/bin/bash
# Run this script once on the login node before submitting any jobs.
# It clones the repo, installs dependencies, and downloads model weights.
#
# Usage:
#   bash scripts/setup_cluster.sh

set -e

REPO=/scratch/vvjumle/clip-food-good
MODELS=/scratch/vvjumle/models
LOGS=/scratch/vvjumle/logs

echo "=== Setting up clip-food-good on HPC ==="

# Clone or pull repo
if [ -d "$REPO/.git" ]; then
    echo "Repo already exists, pulling latest..."
    cd $REPO && git pull
else
    echo "Cloning repo..."
    git clone https://github.com/vedant-jumle/clip-food-good $REPO
fi

# Create directories
mkdir -p $MODELS $LOGS
mkdir -p $REPO/outputs

# Load modules and activate env
module load 2025
module load miniconda3
conda activate CV

# Install dependencies
echo "Installing requirements..."
pip install -r $REPO/requirements.txt

# Download model weights (no internet during jobs)
echo "Downloading CLIP model weights to $MODELS ..."
TORCH_HOME=$MODELS python $REPO/scripts/download_models.py

echo "=== Setup complete ==="
echo "Submit jobs with: sbatch $REPO/scripts/job_experiment<N>.sh"
