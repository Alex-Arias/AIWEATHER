#!/usr/bin/env python3
"""
Operational GraphCast Forecast
==============================

Runs an operational GraphCast forecast using Earth2Studio.

Author: Alex Arias
"""

import os
import time
from datetime import datetime

import torch
import jax
import earth2studio

from earth2studio.run import deterministic
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import GraphCastOperational

# ============================================================
# Forecast Configuration
# ============================================================

INIT_TIME = datetime(2026, 7, 24, 0)

NSTEPS = 40                 # 12 x 6 h = 240 hours

OUTPUT_DIR = "outputs"

timestamp = INIT_TIME.strftime("%Y%m%d_%H")

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    f"graphcast_{timestamp}_240hGeneviere.zarr",
)

# ============================================================
# Environment
# ============================================================

print("=" * 70)
print("Earth2Studio Operational GraphCast")
print("=" * 70)

print(f"Earth2Studio version : {earth2studio.__version__}")
print(f"Python               : {os.sys.version.split()[0]}")
print(f"CUDA available       : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print(f"GPU                  : {torch.cuda.get_device_name(0)}")
print(f"GPU count            : {torch.cuda.device_count()}")
print(f"JAX backend          : {jax.default_backend()}")
print(f"JAX devices          : {jax.devices()}")

# ============================================================
# Output directory
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load GraphCast
# ============================================================

print()
print("=" * 70)
print("Loading GraphCastOperational")
print("=" * 70)

package = GraphCastOperational.load_default_package()

print("Checkpoint package loaded.")

model = GraphCastOperational.load_model(package)

print("GraphCast model loaded successfully.")

# ============================================================
# Initialize Data Source
# ============================================================

print()
print("=" * 70)
print("Initializing GFS")
print("=" * 70)

data = GFS()

print("GFS initialized.")

# ============================================================
# Output Backend
# ============================================================

io = ZarrBackend(OUTPUT_FILE)

# ============================================================
# Forecast Summary
# ============================================================

print()
print("=" * 70)
print("Forecast Configuration")
print("=" * 70)

print(f"Initialization time : {INIT_TIME}")
print(f"Forecast length     : {NSTEPS * 6} hours")
print(f"Forecast steps      : {NSTEPS}")
print(f"Output              : {OUTPUT_FILE}")

# ============================================================
# Run Forecast
# ============================================================

print()
print("=" * 70)
print("Running GraphCast Forecast")
print("=" * 70)

start = time.time()

try:

    deterministic(
        time=[INIT_TIME],
        nsteps=NSTEPS,
        prognostic=model,
        data=data,
        io=io,
    )

except Exception as e:

    print()
    print("=" * 70)
    print("FORECAST FAILED")
    print("=" * 70)
    print(type(e).__name__)
    print(e)
    raise

elapsed = time.time() - start

# ============================================================
# Finished
# ============================================================

print()
print("=" * 70)
print("Forecast completed successfully")
print("=" * 70)

print(f"Elapsed time : {elapsed:.2f} seconds")
print(f"Output file  : {OUTPUT_FILE}")