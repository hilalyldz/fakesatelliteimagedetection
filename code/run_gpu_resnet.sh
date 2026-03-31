#!/bin/bash
#SBATCH --job-name=resnet
#SBATCH --output=results/resnet_%A_%a.out
#SBATCH --error=results/resnet_%A_%a.err
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00

source ~/.bashrc
conda activate resnet312

FEATURE=fft

python -u run_training.py --dataset satellite --feature $FEATURE --gpu-id 0

