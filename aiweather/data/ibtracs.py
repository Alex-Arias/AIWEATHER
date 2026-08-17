"""
IBTrACS dataset acquisition and cache management.

This module manages local copies of NOAA NCEI IBTrACS CSV
datasets. Parsing of IBTrACS storm records remains in
``aiweather.verification.ibtracs``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
import shutil
import tempfile


IBTRACS_VERSION = "v04r01"

IBTRACS_BASE_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    f"{IBTRACS_VERSION}/access/csv"
)

SUPPORTED_BASINS = {
    "EP",
    "NA",
    "NI",
    "SI",
    "SP",
    "WP",
}


def ibtracs_filename(
    basin: str,
) -> str:
    """
    Return the NOAA IBTrACS CSV filename for a basin.
    """
    if not isinstance(
        basin,
        str,
    ):
        raise TypeError(
            "basin must be a string."
        )

    basin = basin.upper()

    if basin not in SUPPORTED_BASINS:
        raise ValueError(
            "unsupported IBTrACS basin: "
            f"{basin!r}"
        )

    return (
        f"ibtracs.{basin}.list."
        f"{IBTRACS_VERSION}.csv"
    )


def ibtracs_url(
    basin: str,
) -> str:
    """
    Return the NOAA download URL for a basin CSV.
    """
    return (
        f"{IBTRACS_BASE_URL}/"
        f"{ibtracs_filename(basin)}"
    )


def _file_age_hours(
    path: Path,
    *,
    now: datetime | None = None,
) -> float:
    """
    Return local file age in hours.
    """
    if now is None:
        now = datetime.now(
            timezone.utc
        )

    if now.tzinfo is None:
        raise ValueError(
            "now must be timezone-aware."
        )

    modified = datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=timezone.utc,
    )

    age = (
        now - modified
    ).total_seconds() / 3600.0

    return max(
        0.0,
        age,
    )


def ibtracs_cache_is_fresh(
    path: str | Path,
    *,
    max_age_hours: float = 48.0,
    now: datetime | None = None,
) -> bool:
    """
    Return True when a cached IBTrACS file is sufficiently recent.
    """
    path = Path(
        path
    )

    if max_age_hours <= 0.0:
        raise ValueError(
            "max_age_hours must be positive."
        )

    if not path.is_file():
        return False

    if path.stat().st_size <= 0:
        return False

    return (
        _file_age_hours(
            path,
            now=now,
        )
        <= max_age_hours
    )


def download_ibtracs_dataset(
    basin: str,
    destination: str | Path,
) -> Path:
    """
    Download one NOAA IBTrACS basin CSV atomically.

    The existing destination is replaced only after a complete
    temporary download succeeds.
    """
    destination = Path(
        destination
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    url = ibtracs_url(
        basin
    )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=(
                destination.name
                + "."
            ),
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:

            temporary_path = Path(
                temporary_file.name
            )

            with urlopen(
                url,
                timeout=120,
            ) as response:
                shutil.copyfileobj(
                    response,
                    temporary_file,
                )

        if (
            temporary_path.stat().st_size
            <= 0
        ):
            raise RuntimeError(
                "downloaded IBTrACS file is empty."
            )

        temporary_path.replace(
            destination
        )

    except Exception:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()

        raise

    return destination


def ensure_ibtracs_dataset(
    *,
    basin: str = "EP",
    cache_dir: str | Path = (
        "data/verification/ibtracs"
    ),
    max_age_hours: float = 48.0,
    force_update: bool = False,
) -> Path:
    """
    Ensure that a usable local IBTrACS basin CSV is available.

    A fresh cached file is reused unless ``force_update=True``.
    Missing or stale datasets are downloaded from NOAA NCEI.

    Parameters
    ----------
    basin
        IBTrACS basin code.

    cache_dir
        Local cache directory.

    max_age_hours
        Maximum acceptable cache age.

    force_update
        Download even when the cached file is still fresh.

    Returns
    -------
    pathlib.Path
        Local path to the cached IBTrACS CSV.
    """
    if not isinstance(
        force_update,
        bool,
    ):
        raise TypeError(
            "force_update must be a boolean."
        )

    if max_age_hours <= 0.0:
        raise ValueError(
            "max_age_hours must be positive."
        )

    filename = ibtracs_filename(
        basin
    )

    destination = (
        Path(cache_dir)
        / filename
    )

    if (
        not force_update
        and ibtracs_cache_is_fresh(
            destination,
            max_age_hours=max_age_hours,
        )
    ):
        return destination

    return download_ibtracs_dataset(
        basin,
        destination,
    )