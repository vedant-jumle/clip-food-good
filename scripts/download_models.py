"""Run this script once on the login node before submitting any jobs.
It downloads and caches the OpenCLIP ViT-B/32 model weights so jobs
can run without internet access.

Usage:
    TORCH_HOME=/scratch/vvjumle/models python scripts/download_models.py
"""
from __future__ import annotations

import os

import open_clip

cache_dir = os.environ.get("TORCH_HOME", "/scratch/vvjumle/models")
os.environ["TORCH_HOME"] = cache_dir
print(f"Downloading ViT-B/32 (openai) to cache: {cache_dir}")

model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
print("Done.")
