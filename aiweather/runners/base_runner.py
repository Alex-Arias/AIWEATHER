"""
Base runner class for all AIWeather models.
"""

from __future__ import annotations

import time


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

            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA device not available."
                )

            print(
                f"Using GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )

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

        from earth2studio.io import ZarrBackend

        from aiweather.output import OutputManager

        if request.output_path is None:
            output_path = OutputManager.build_output_path(
                request
            )
        else:
            output_path = request.output_path

        request.output_path = output_path

        self.io = ZarrBackend(
            output_path
        )

        print()
        print(f"Output: {output_path}")

        return output_path




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

        self.build_output(request)

        self.run_forecast(request)

        self.cleanup()

        elapsed = time.time() - t0

        #print()
        #print(f"Finished in {elapsed:.1f} seconds")

        return None
