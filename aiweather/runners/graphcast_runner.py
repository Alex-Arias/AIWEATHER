"""
GraphCast operational runner.
"""

from __future__ import annotations

from earth2studio.models.px import GraphCastOperational

from .base_runner import BaseRunner

from earth2studio.data import GFS

import torch




class GraphCastRunner(BaseRunner):

    def __init__(self):

        self.model = None

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    def check_device(
        self,
        request,
    ):

        print()

        print("Checking execution device...")
        
        if request.device == "cuda":

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device not available.")

            print(f"Using GPU: {torch.cuda.get_device_name(0)}")

        elif request.device == "cpu":

            print("Using CPU.")

        else:

            raise ValueError(f"Unknown device: {request.device}")






    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    def load_model(self):

        print()

        print("Loading GraphCastOperational...")

        package = GraphCastOperational.load_default_package()

        self.model = GraphCastOperational.load_model(package)

        print("GraphCast loaded.")

    
    # ---------------------------------------------------------
    # Data  
    # ---------------------------------------------------------

    def load_data(
        self,
    ):

        print()

        print("Initializing GFS...")

       return GFS()
    
    
    # ---------------------------------------------------------
    # Forecast
    # ---------------------------------------------------------

    def run(
        self,
        request,
    ):

        self.check_device(request)

        if self.model is None:
            self.load_model()

        data = self.load_data()

        print("Data source initialized.")

        raise NotImplementedError