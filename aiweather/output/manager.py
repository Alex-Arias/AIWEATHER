"""
AIWeather Output Manager.

Responsible for creating standardized forecast output directories.
"""

from __future__ import annotations

from pathlib import Path


class OutputManager:
    """
    Creates standardized output paths for AIWeather forecasts.
    """

    BASE_OUTPUT = Path("outputs")

    @classmethod
    def build_output_path(cls, request):
        """
        Build the forecast output path.

        Example
        -------
        outputs/
            graphcast/
                20260724T000000/
                    forecast.zarr
        """

        init = (
            request.init_time
            .replace("-", "")
            .replace(":", "")
        )

        output_dir = (
            cls.BASE_OUTPUT
            / request.model.lower()
            / init
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir / "forecast.zarr"