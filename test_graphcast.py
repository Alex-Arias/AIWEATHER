"""
GraphCast integration test for AIWeather.

This script verifies that:

1. GraphCast loads successfully.
2. AIWeather preprocessing succeeds.
3. GraphCast creates a forecast iterator.
4. The first forecast can be retrieved.
5. The returned objects are inspected.

Usage:

    python test_graphcast.py
"""

import traceback

import xarray as xr

from aiweather.models.graphcast import GraphCastModel


# ---------------------------------------------------------------------
# Minimal synthetic dataset
# ---------------------------------------------------------------------

ds = xr.Dataset(
    data_vars={
        "t2m": (("time", "lat", "lon"), [[[280.0]]]),
        "msl": (("time", "lat", "lon"), [[[101325.0]]]),
        "u10m": (("time", "lat", "lon"), [[[2.0]]]),
        "v10m": (("time", "lat", "lon"), [[[1.0]]]),
    },
    coords={
        "time": [0],
        "lat": [45.0],
        "lon": [10.0],
    },
)


print("=" * 70)
print("AIWeather GraphCast Integration Test")
print("=" * 70)

print("\nCreating GraphCast model...\n")

model = GraphCastModel()

try:

    result = model.run(
        ds,
        lead_time=6,
    )

    print("\n")
    print("=" * 70)
    print("GraphCast returned successfully")
    print("=" * 70)

    print()

    print("Returned object type:")
    print(type(result))

    print()

    #
    # If the adapter returns a tuple
    #
    if isinstance(result, tuple):

        print(f"Tuple contains {len(result)} objects\n")

        for i, obj in enumerate(result):

            print(f"Object {i}")
            print("-" * 40)

            print(type(obj))

            if hasattr(obj, "shape"):
                print("Shape:")
                print(obj.shape)

            print()

            print(obj)

            print()

    else:

        print(result)

except Exception:

    print("\n")
    print("=" * 70)
    print("EXCEPTION")
    print("=" * 70)

    traceback.print_exc()