#!/bin/bash
#SBATCH --job-name=graphcast
#SBATCH --partition=oc_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/graphcast_%j.out
#SBATCH --error=logs/graphcast_%j.err

module purge

source ~/.bashrc
conda activate earth2studio

cd /LUSTRE/ariasv/AIWeather/TestGCast

mkdir -p logs
mkdir -p outputs

python graphcast_forecast.py 2>&1 | tee logs/graphcast_${SLURM_JOB_ID}.log