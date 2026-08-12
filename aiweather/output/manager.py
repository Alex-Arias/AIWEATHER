"""
AIWeather Output Manager.

Responsible for creating standardized forecast output directories
and persistent forecast metadata.
"""

from __future__ import annotations

from pathlib import Path

import zarr


class OutputManager:
    """
    Creates standardized output paths for AIWeather forecasts.
    """

    BASE_OUTPUT = Path("outputs")

    @staticmethod
    def _normalized_init_time(request) -> str:
        return (
            request.init_time
            .replace("-", "")
            .replace(":", "")
        )

    @classmethod
    def build_forecast_id(cls, request) -> str:
        """
        Build a semantic forecast identifier.

        Example
        -------
        graphcast_gfs_20260724T000000_240h
        """
        init = cls._normalized_init_time(
            request
        )

        return (
            f"{request.model.lower()}_"
            f"{request.datasource.lower()}_"
            f"{init}_"
            f"{int(request.lead_time)}h"
        )

    @classmethod
    def build_output_path(cls, request):
        """
        Build the canonical forecast output path.

        Example
        -------
        outputs/
            graphcast/
                20260724T000000/
                    forecast.zarr
        """
        init = cls._normalized_init_time(
            request
        )

        output_dir = (
            cls.BASE_OUTPUT
            / request.model.lower()
            / init
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return output_dir / "forecast.zarr"

    @classmethod
    def write_forecast_metadata(
        cls,
        output_path,
        request,
    ) -> None:
        """
        Persist AIWeather run metadata in a Zarr store.
        """

        group = zarr.open_group(
            str(output_path),
            mode="a",
        )

        group.attrs.update(
            {
                "aiweather_model": request.model.lower(),
                "aiweather_datasource": request.datasource.lower(),
                "aiweather_initialization_time": request.init_time,
                "aiweather_lead_time_hours": int(
                    request.lead_time
                ),
                "aiweather_forecast_id": cls.build_forecast_id(
                    request
                ),
                "aiweather_backend": "earth2studio",
            }
        )

        zarr.consolidate_metadata(
            str(output_path)
        )
