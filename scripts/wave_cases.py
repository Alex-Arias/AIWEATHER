"""
Configuration for AIFS2 tropical-cyclone wave cases.

Case-specific metadata are kept here so that the wave-analysis
and plotting scripts can use the same processing methodology
for every storm.
"""


CASES = {
    "elida": {
        "storm_name": "Elida",
        "init": "20260714T120000",
        "case_id": "elida_20260714T120000",
        "center_source": "wuduan",
    },

    "fausto": {
        "storm_name": "Fausto",
        "init": "20260719T000000",
        "case_id": "fausto_20260719T000000",
        "center_source": "wuduan",
    },

    "genevieve": {
        "storm_name": "Genevieve",
        "init": "20260724T000000",
        "case_id": "genevieve_20260724T000000",
        "center_source": "wuduan",
    },

    "hernan": {
        "storm_name": "Hernan",
        "init": "20260811T000000",
        "case_id": "hernan_20260811T000000",
        "center_source": "wuduan",
    },

    "douglas": {
        "storm_name": "Douglas",
        "init": "20260629T000000",
        "case_id": "douglas_20260629T000000",
        "center_source": "ibtracs",
        "center_path": (
            "results/verification/"
            "aifs2_douglas_20260629T000000/"
            "ibtracs.csv"
        ),
    },

    "iselle": {
        "storm_name": "Iselle",
        "init": "20260821T180000",
        "case_id": "iselle_20260821T180000",
        "center_source": "ibtracs",
        "center_path": (
            "results/verification/"
            "aifs2_iselle_20260821T180000/"
            "ibtracs.csv"
        ),
    },

    "karina": {
        "storm_name": "Karina",
        "init": "20260829T120000",
        "case_id": "karina_20260829T120000",
        "center_source": "operational",
        "center_path": (
            "results/operational/"
            "karina_20260829T120000/"
            "aifs2_track.csv"
        ),
        "forecast_path": (
            "outputs/aifs2/"
            "20260829T120000/"
            "forecast_azure.zarr"
        ),
        "wave_lead_times": [
            0,
            24,
            48,
            72,
            96,
            120,
            144,
            168,
            192,
            216,
            240,
        ],
        "swh_max": 10.0,
        "mwp_max": 16.0,
        "wind_max": 30.0,
    },
}


def get_case(storm):
    """
    Return configuration for one storm.
    """

    key = storm.lower()

    if key not in CASES:
        available = ", ".join(
            sorted(CASES)
        )

        raise ValueError(
            f"Unknown storm {storm!r}. "
            f"Available storms: {available}"
        )

    return CASES[key]
