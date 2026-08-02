"""
Base runner class for all AIWeather models.
"""

from __future__ import annotations

import os
import time

import torch
from earth2studio.io import ZarrBackend


class BaseRunner:

    def __init__(self):

        self.model = None
        self.data = None
        self.io = None

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    def check_device(self, request):

        print()
        print("Checking execution device...")

        if request.device == "cuda":

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device not available.")

            print(f"Using GPU: {torch.cuda.get_device_name(0)}")

        elif request.device == "cpu":

            print("Using CPU.")

        else:

            raise ValueError(
                f"Unknown device: {request.device}"
            )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    def build_output(self, request):

        output_dir = request.output_path

        if output_dir is None:
            output_dir = "outputs"

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        filename = (
            f"{request.model}_"
            f"{request.datasource}_"
            f"{request.init_time.replace(':','').replace('-','')}_"
            f"{request.lead_time}h.zarr"
        )

        output_file = os.path.join(
            output_dir,
            filename,
        )

        self.io = ZarrBackend(output_file)

        print()
        print(f"Output: {output_file}")

        return output_file

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    def cleanup(self):

        pass

    # ---------------------------------------------------------
    # Main execution
    # ---------------------------------------------------------

    def run(self, request):

        t0 = time.time()

        self.check_device(request)

        if self.model is None:
            self.load_model()

        if self.data is None:
            self.load_data()

        output = self.build_output(request)

        self.run_forecast(request)

        self.cleanup()

        elapsed = time.time() - t0

        print()
        print(f"Finished in {elapsed:.1f} seconds")

        return output