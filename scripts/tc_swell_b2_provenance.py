"""Phase B2 scientific provenance utilities.

This module records available evidence without inventing
historical processing lineage.

It does not modify scientific products or establish physical
attribution of tropical-cyclone-generated swell.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


SCHEMA_VERSION = "b2-provenance-1"


def sha256_file(path):
    """Calculate the SHA256 digest of a regular file."""
    path = Path(path)

    if not path.is_file() or path.is_symlink():
        raise ValueError(
            f"Expected regular file: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def file_record(path, *, expected_sha256=None):
    """Record file identity and optionally enforce its checksum."""
    path = Path(path)

    actual = sha256_file(path)

    if (
        expected_sha256 is not None
        and actual != expected_sha256
    ):
        raise ValueError(
            f"SHA256 mismatch: {path}"
        )

    return {
        "path": str(path),
        "sha256": actual,
        "size_bytes": path.stat().st_size,
    }


def validate_forecast_identity(
    forecast_provenance,
    *,
    storm,
    storm_id,
    initialization,
    model="aifs2",
):
    """Validate explicit forecast identity metadata."""
    forecast = forecast_provenance["forecast"]
    storm_metadata = forecast_provenance["storm"]

    expected_time = datetime.strptime(
        initialization,
        "%Y%m%dT%H%M%S",
    )

    if expected_time.strftime(
        "%Y%m%dT%H%M%S"
    ) != initialization:
        raise ValueError(
            "Noncanonical initialization"
        )

    recorded_time = datetime.fromisoformat(
        forecast["initialization_time"]
    )

    if recorded_time.tzinfo is not None:
        raise ValueError(
            "Timezone-aware forecast initialization "
            "requires explicit normalization"
        )

    if recorded_time != expected_time:
        raise ValueError(
            "Forecast initialization mismatch"
        )

    if forecast["model_name"].lower() != model.lower():
        raise ValueError(
            "Forecast model mismatch"
        )

    if storm_metadata["name"].lower() != storm.lower():
        raise ValueError(
            "Forecast storm mismatch"
        )

    if storm_metadata["id"].upper() != storm_id.upper():
        raise ValueError(
            "Forecast storm identifier mismatch"
        )


def validate_tracker_identity(
    tracker_provenance,
    *,
    storm,
    initialization,
    tracker,
    model="aifs2",
):
    """Validate explicit tracker identity metadata."""
    expected_time = datetime.strptime(
        initialization,
        "%Y%m%dT%H%M%S",
    )

    recorded_time = datetime.fromisoformat(
        tracker_provenance["initialization_time"]
    )

    if recorded_time.tzinfo is not None:
        raise ValueError(
            "Timezone-aware tracker initialization "
            "requires explicit normalization"
        )

    if recorded_time != expected_time:
        raise ValueError(
            "Tracker initialization mismatch"
        )

    for field, expected in (
        ("storm_name", storm),
        ("tracker", tracker),
        ("model", model),
    ):
        actual = tracker_provenance[field]

        if str(actual).lower() != expected.lower():
            raise ValueError(
                f"Tracker provenance mismatch: {field}"
            )


def build_manifest(
    *,
    storm,
    storm_id,
    initialization,
    tracker,
    forecast_provenance_path,
    tracker_provenance_path,
    scientific_inputs,
    processing_scripts,
    retrospective=True,
    expected_scientific_sha256=None,
    derived_products=None,
    expected_derived_sha256=None,
):
    """Build a provenance record from explicit available evidence.

    scientific_inputs and processing_scripts map logical names
    to existing paths. All file checksums are computed directly.

    A retrospective record does not prove that these exact
    files were used in the historical execution.
    """
    if not retrospective:
        raise ValueError(
            "Execution-time provenance is not implemented; "
            "use retrospective=True"
        )

    forecast_path = Path(forecast_provenance_path)
    tracker_path = Path(tracker_provenance_path)

    forecast_metadata = json.loads(
        forecast_path.read_text()
    )

    tracker_metadata = json.loads(
        tracker_path.read_text()
    )

    validate_forecast_identity(
        forecast_metadata,
        storm=storm,
        storm_id=storm_id,
        initialization=initialization,
    )

    validate_tracker_identity(
        tracker_metadata,
        storm=storm,
        initialization=initialization,
        tracker=tracker,
    )

    native_forecast_path = (
        forecast_metadata["forecast"]["path"]
    )

    tracker_forecast_path = (
        tracker_metadata["forecast_path"]
    )

    if native_forecast_path != tracker_forecast_path:
        raise ValueError(
            "Native and tracker forecast paths disagree"
        )

    if expected_scientific_sha256 is not None:
        if not isinstance(expected_scientific_sha256, dict):
            raise ValueError(
                "Expected scientific checksums must be a dictionary"
            )

        expected_names = set(scientific_inputs)
        supplied_names = set(expected_scientific_sha256)

        if expected_names != supplied_names:
            raise ValueError(
                "Accepted scientific checksum inventory "
                "does not match scientific inputs"
            )

        for name, expected_sha in expected_scientific_sha256.items():
            if (
                not isinstance(expected_sha, str)
                or len(expected_sha) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected_sha
                )
            ):
                raise ValueError(
                    f"Invalid accepted SHA256 for {name}"
                )

    inputs = {
        name: file_record(
            path,
            expected_sha256=(
                expected_scientific_sha256[name]
                if expected_scientific_sha256 is not None
                else None
            ),
        )
        for name, path in sorted(
            scientific_inputs.items()
        )
    }

    scripts = {
        name: file_record(path)
        for name, path in sorted(
            processing_scripts.items()
        )
    }

    if derived_products is None:
        if expected_derived_sha256 is not None:
            raise ValueError(
                "Derived checksum inventory supplied "
                "without derived products"
            )
        derived = None
    else:
        if not isinstance(derived_products, dict):
            raise ValueError(
                "Derived products must be a dictionary"
            )

        if not derived_products:
            raise ValueError(
                "Derived products inventory cannot be empty"
            )

        if not isinstance(expected_derived_sha256, dict):
            raise ValueError(
                "Accepted derived checksums must be a dictionary"
            )

        if set(derived_products) != set(expected_derived_sha256):
            raise ValueError(
                "Accepted derived checksum inventory "
                "does not match derived products"
            )

        for name, expected_sha in expected_derived_sha256.items():
            if (
                not isinstance(expected_sha, str)
                or len(expected_sha) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected_sha
                )
            ):
                raise ValueError(
                    f"Invalid accepted derived SHA256 for {name}"
                )

        overlap = (
            set(derived_products)
            & set(scientific_inputs)
        )

        if overlap:
            raise ValueError(
                "Derived products and scientific inputs "
                "must use distinct logical names"
            )

        derived_paths = {
            str(Path(path).resolve(strict=True))
            for path in derived_products.values()
        }

        input_paths = {
            str(Path(path).resolve(strict=True))
            for path in scientific_inputs.values()
        }

        if derived_paths & input_paths:
            raise ValueError(
                "Derived products cannot also be "
                "scientific input files"
            )

        derived = {
            name: file_record(
                path,
                expected_sha256=expected_derived_sha256[name],
            )
            for name, path in sorted(
                derived_products.items()
            )
        }

    manifest = {
        "schema_version": (
            "b2-provenance-2"
            if derived is not None
            else SCHEMA_VERSION
        ),
        "record_type": (
            "retrospective_scientific_provenance"
            if retrospective
            else "execution_scientific_provenance"
        ),
        "experiment": {
            "storm": storm,
            "storm_id": storm_id,
            "initialization": initialization,
            "model": "aifs2",
            "tracker": tracker,
        },
        "forecast": {
            "metadata": file_record(forecast_path),
            "forecast_id": (
                forecast_metadata["forecast"]["forecast_id"]
            ),
            "forecast_path": (
                forecast_metadata["forecast"]["path"]
            ),
            "datasource": (
                forecast_metadata["forecast"]["datasource"]
            ),
            "datasource_source": (
                forecast_metadata["forecast"]["datasource_source"]
            ),
        },
        "tracking": {
            "metadata": file_record(tracker_path),
            "tracker_path_id": (
                tracker_metadata["tracker_path_id"]
            ),
            "selection_method": (
                tracker_metadata["selection_method"]
            ),
        },
        "scientific_inputs": inputs,
        "processing_scripts": scripts,
        "verification": {
            "forecast_identity": "metadata_validated",
            "tracker_identity": "metadata_validated",
            "input_file_checksums": (
                "accepted_inventory_verified"
                if expected_scientific_sha256 is not None
                else "computed"
            ),
            "processing_script_checksums": "computed",
            "historical_spectral_lineage": "unverified",
            "forecast_zarr_content": "not_verified",
            "physical_tc_swell_attribution": "not_established",
        },
    }

    if derived is not None:
        manifest["derived_products"] = derived
        manifest["verification"][
            "derived_product_checksums"
        ] = "accepted_inventory_verified"

    return manifest


def write_manifest_exclusive(manifest, output):
    """Write JSON exclusively without replacing existing files."""
    output = Path(output)

    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"Manifest already exists: {output}"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = None

    try:
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=".b2_manifest_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)

            json.dump(
                manifest,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )

            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.link(temporary, output)

    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
