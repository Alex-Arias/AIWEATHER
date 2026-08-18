"""
AIWeather Output Manager.

Responsible for creating standardized forecast output directories,
verification result directories, and persistent forecast metadata.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import zarr


class OutputManager:
    """
    Creates standardized output paths for AIWeather forecasts
    and verification products.
    """

    BASE_OUTPUT = Path("outputs")
    BASE_RESULTS = Path("results")

    @staticmethod
    def _normalized_init_time(
        request,
    ) -> str:
        """
        Normalize a forecast-request initialization time string.
        """
        return (
            request.init_time
            .replace("-", "")
            .replace(":", "")
        )

    @staticmethod
    def _format_initialization_time(
        initialization_time: datetime,
    ) -> str:
        """
        Format a forecast initialization time for canonical paths.

        Example
        -------
        2026-07-24 00:00:00
            -> 20260724T000000
        """
        if not isinstance(
            initialization_time,
            datetime,
        ):
            raise TypeError(
                "initialization_time must be "
                "a datetime."
            )

        return initialization_time.strftime(
            "%Y%m%dT%H%M%S"
        )

    @classmethod
    def build_forecast_id(
        cls,
        request,
    ) -> str:
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
    def build_output_path(
        cls,
        request,
    ) -> Path:
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

        return (
            output_dir
            / "forecast.zarr"
        )

    @classmethod
    def build_verification_case_name(
        cls,
        *,
        sid: str,
        initialization_time: datetime,
    ) -> str:
        """
        Build a canonical tropical-cyclone verification case name.

        Parameters
        ----------
        sid
            IBTrACS storm identifier.

        initialization_time
            Forecast initialization time.

        Returns
        -------
        str
            Canonical verification case name.

        Example
        -------
        2026204N08267_20260724T000000
        """
        if not isinstance(
            sid,
            str,
        ):
            raise TypeError(
                "sid must be a string."
            )

        sid = sid.strip()

        if not sid:
            raise ValueError(
                "sid cannot be empty."
            )

        init = (
            cls._format_initialization_time(
                initialization_time
            )
        )

        return (
            f"{sid}_{init}"
        )

    @classmethod
    def build_verification_output_dir(
        cls,
        *,
        sid: str,
        initialization_time: datetime,
        create: bool = True,
    ) -> Path:
        """
        Build the canonical verification output directory.

        Parameters
        ----------
        sid
            IBTrACS storm identifier.

        initialization_time
            Forecast initialization time.

        create
            Create the directory when True.

        Returns
        -------
        pathlib.Path
            Canonical verification result directory.

        Example
        -------
        results/
            verification/
                2026204N08267_20260724T000000/
        """
        if not isinstance(
            create,
            bool,
        ):
            raise TypeError(
                "create must be a boolean."
            )

        case_name = (
            cls.build_verification_case_name(
                sid=sid,
                initialization_time=(
                    initialization_time
                ),
            )
        )

        output_dir = (
            cls.BASE_RESULTS
            / "verification"
            / case_name
        )

        if create:
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        return output_dir

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
                "aiweather_model":
                    request.model.lower(),
                "aiweather_datasource":
                    request.datasource.lower(),
                "aiweather_initialization_time":
                    request.init_time,
                "aiweather_lead_time_hours":
                    int(
                        request.lead_time
                    ),
                "aiweather_forecast_id":
                    cls.build_forecast_id(
                        request
                    ),
                "aiweather_backend":
                    "earth2studio",
            }
        )

        zarr.consolidate_metadata(
            str(output_path)
        )