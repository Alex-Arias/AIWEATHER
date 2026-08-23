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
    },

    "fausto": {
        "storm_name": "Fausto",
        "init": "20260719T000000",
        "case_id": "fausto_20260719T000000",
    },

    "genevieve": {
        "storm_name": "Genevieve",
        "init": "20260724T000000",
        "case_id": "genevieve_20260724T000000",
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
