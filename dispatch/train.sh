#!/bin/bash
#SBATCH --job-name=train_cw
#SBATCH --cpus-per-task=64
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/train_cw_%A_%a.out
#SBATCH --error=logs/train_cw_%A_%a.err

module load uv
uv run main.py "$@"