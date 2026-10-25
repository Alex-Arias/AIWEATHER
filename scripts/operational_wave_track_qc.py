"""Optional Native-track QC for operational wave diagnostics.

This module does not modify forecast data or original track files.
"""

import pandas as pd


REQUIRED_QC_COLUMNS = {
    "model",
    "cycle_time",
    "lead_time_hours",
    "qc_action",
}

EXCLUSION_ACTION = "exclude_from_derived_variability"


def filter_native_track(
    track: pd.DataFrame,
    qc: pd.DataFrame | None = None,
    *,
    model: str = "aifs2",
    init: str,
) -> pd.DataFrame:
    """Return a track copy with applicable QC exclusions removed."""

    if "lead_time_hours" not in track.columns:
        raise ValueError(
            "Track is missing lead_time_hours"
        )

    result = track.copy()

    if qc is None:
        return result

    missing = REQUIRED_QC_COLUMNS - set(qc.columns)

    if missing:
        raise ValueError(
            "QC table missing columns: "
            + ", ".join(sorted(missing))
        )

    cycle_time = pd.to_datetime(
        init,
        format="%Y%m%dT%H%M%S",
        errors="raise",
    )

    qc_times = pd.to_datetime(
        qc["cycle_time"],
        errors="raise",
    )

    relevant = qc.loc[
        qc["model"].astype(str).str.lower().eq(model.lower())
        & qc_times.eq(cycle_time)
        & qc["qc_action"].eq(EXCLUSION_ACTION)
    ].copy()

    if relevant.empty:
        return result

    leads = pd.to_numeric(
        relevant["lead_time_hours"],
        errors="raise",
    )

    if leads.isna().any():
        raise ValueError("QC contains missing lead times")

    if not (leads % 1 == 0).all():
        raise ValueError("QC contains noninteger lead times")

    if leads.duplicated().any():
        raise ValueError(
            "Duplicate QC exclusions for the same lead"
        )

    track_leads = pd.to_numeric(
        result["lead_time_hours"],
        errors="raise",
    )

    return result.loc[
        ~track_leads.isin(leads)
    ].copy().reset_index(drop=True)
