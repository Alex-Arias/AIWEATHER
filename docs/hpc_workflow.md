# HPC workflow

AIWeather development and production runs are performed from the repository:

```bash
cd /LUSTRE/ariasv/AIWeather
conda activate aiweather17_aifs2
```

## Forecast runs

Forecast generation is the expensive part of the workflow and should be run on an appropriate compute/GPU node rather than on a login node.

A representative interactive allocation is:

```bash
srun \
    -p oc_gpu \
    --gres=gpu:1 \
    --cpus-per-task=64 \
    --mem=128G \
    --time=04:00:00 \
    --pty bash
```

Cluster resource requests should be adjusted to local policy and model requirements.

If the CUDA environment needs to be set explicitly:

```bash
export CUDA_HOME=/data/apps/nvidia/hpc_sdk_2024/Linux_x86_64/24.5/cuda/12.4
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
```

## Verification and plotting

TC verification can be run with:

```text
--device cpu
```

A GPU is therefore not required merely to generate the verification tables and plots. Use a compute node for substantial processing and avoid running long or memory-intensive jobs on the login node.

## Pre-run checks

```bash
which python
python --version
which aiweather
aiweather --version
nvidia-smi
```

For forecast stores:

```bash
find outputs/aifs2 -mindepth 2 -maxdepth 2 -type d -name forecast.zarr | sort
find outputs/graphcast -mindepth 2 -maxdepth 2 -type d -name forecast.zarr | sort
```
