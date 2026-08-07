#!/bin/bash
#SBATCH --job-name=graphcast
#SBATCH --partition=oc_gpu
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

#SBATCH --output=logs/graphcast_%j.out
#SBATCH --error=logs/graphcast_%j.err

##############################################################################
# Project directory
##############################################################################

cd /LUSTRE/ariasv/TestGCast || exit 1

mkdir -p logs outputs

##############################################################################
# Environment
##############################################################################

module purge
module load cuda12.4
module load conda/conda-2026

source /data/apps/conda/2026/miniconda3/etc/profile.d/conda.sh

conda activate earth2studio

# Force bash to forget cached commands
hash -r

echo "======================================================"
echo "Host            : $(hostname)"
echo "Directory       : $(pwd)"
echo "Date            : $(date)"
echo "======================================================"

echo
echo "Conda"
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"

echo
echo "PATH"
echo "$PATH"

echo
echo "System python"
which python
python --version

echo
echo "Environment python"
$CONDA_PREFIX/bin/python --version

echo
echo "Testing imports with environment python"

$CONDA_PREFIX/bin/python - <<EOF
import sys
print("Executable :", sys.executable)

import torch
print("Torch      :", torch.__version__)
print("CUDA avail :", torch.cuda.is_available())

import jax
print("JAX        :", jax.__version__)
print("Devices    :", jax.devices())

import earth2studio
print("Earth2Studio:", earth2studio.__version__)
EOF

echo
echo "Running GraphCast..."

$CONDA_PREFIX/bin/python graphcast.py

status=$?

echo
echo "GraphCast exit code: $status"

exit $status