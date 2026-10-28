"""Regression tests for Phase B2 provenance safeguards."""

import copy
import hashlib
import json

import pytest

from scripts.tc_swell_b2_provenance import (
    SCHEMA_VERSION,
    build_manifest,
    file_record,
    sha256_file,
    validate_forecast_identity,
    validate_tracker_identity,
    write_manifest_exclusive,
)


INIT = "20261005T120000"
FORECAST_PATH = "outputs/aifs2/20261005T120000/forecast.zarr"


@pytest.fixture
def sample(tmp_path):
    forecast = {
        "forecast": {
            "initialization_time": "2026-10-05T12:00:00",
            "model_name": "aifs2",
            "forecast_id": "aifs2_ifs_20261005T120000_240h",
            "path": FORECAST_PATH,
            "datasource": "ifs",
            "datasource_source": "azure",
        },
        "storm": {
            "name": "Rachel",
            "id": "EP182026",
        },
    }

    tracker = {
        "initialization_time": "2026-10-05 12:00:00",
        "model": "aifs2",
        "tracker": "wuduan",
        "storm_name": "Rachel",
        "forecast_path": FORECAST_PATH,
        "tracker_path_id": 2,
        "selection_method": "previous_cycle_valid_time_continuity",
    }

    forecast_file = tmp_path / "forecast.json"
    tracker_file = tmp_path / "tracker.json"
    scientific_file = tmp_path / "events.csv"
    script_file = tmp_path / "processing.py"

    forecast_file.write_text(json.dumps(forecast))
    tracker_file.write_text(json.dumps(tracker))
    scientific_file.write_text("sector,band\\nA,h1012\\n")
    script_file.write_text("VALUE = 1\\n")

    return {
        "forecast": forecast,
        "tracker": tracker,
        "forecast_file": forecast_file,
        "tracker_file": tracker_file,
        "scientific_file": scientific_file,
        "script_file": script_file,
    }


def make_manifest(sample, **overrides):
    arguments = {
        "storm": "rachel",
        "storm_id": "EP182026",
        "initialization": INIT,
        "tracker": "wuduan",
        "forecast_provenance_path": sample["forecast_file"],
        "tracker_provenance_path": sample["tracker_file"],
        "scientific_inputs": {
            "events": sample["scientific_file"],
        },
        "processing_scripts": {
            "processor": sample["script_file"],
        },
        "retrospective": True,
    }

    arguments.update(overrides)
    return build_manifest(**arguments)


def test_valid_retrospective_manifest(sample):
    result = make_manifest(sample)

    assert result["schema_version"] == SCHEMA_VERSION
    assert result["record_type"] == (
        "retrospective_scientific_provenance"
    )
    assert result["experiment"]["storm"] == "rachel"
    assert result["experiment"]["tracker"] == "wuduan"

    assert result["verification"][
        "historical_spectral_lineage"
    ] == "unverified"

    assert result["verification"][
        "physical_tc_swell_attribution"
    ] == "not_established"


def test_manifest_deterministic(sample):
    first = make_manifest(sample)
    second = make_manifest(sample)

    assert json.dumps(
        first, sort_keys=True
    ) == json.dumps(
        second, sort_keys=True
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("initialization_time", "2026-10-06T12:00:00"),
        ("model_name", "graphcast"),
    ],
)
def test_forecast_identity_mismatch(sample, field, value):
    metadata = copy.deepcopy(sample["forecast"])
    metadata["forecast"][field] = value

    with pytest.raises(ValueError):
        validate_forecast_identity(
            metadata,
            storm="rachel",
            storm_id="EP182026",
            initialization=INIT,
        )


def test_forecast_storm_mismatch(sample):
    metadata = copy.deepcopy(sample["forecast"])
    metadata["storm"]["name"] = "Trudy"

    with pytest.raises(ValueError):
        validate_forecast_identity(
            metadata,
            storm="rachel",
            storm_id="EP182026",
            initialization=INIT,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("tracker", "vitart"),
        ("model", "graphcast"),
        ("storm_name", "Trudy"),
        ("initialization_time", "2026-10-06 12:00:00"),
    ],
)
def test_tracker_identity_mismatch(sample, field, value):
    metadata = copy.deepcopy(sample["tracker"])
    metadata[field] = value

    with pytest.raises(ValueError):
        validate_tracker_identity(
            metadata,
            storm="rachel",
            initialization=INIT,
            tracker="wuduan",
        )


def test_forecast_path_mismatch(sample):
    metadata = copy.deepcopy(sample["tracker"])
    metadata["forecast_path"] = (
        "outputs/aifs2/other/forecast.zarr"
    )

    sample["tracker_file"].write_text(json.dumps(metadata))

    with pytest.raises(
        ValueError,
        match="forecast paths disagree",
    ):
        make_manifest(sample)


def test_execution_provenance_rejected(sample):
    with pytest.raises(
        ValueError,
        match="Execution-time provenance",
    ):
        make_manifest(sample, retrospective=False)


def test_missing_scientific_input_rejected(sample):
    missing = sample["scientific_file"].parent / "missing.csv"

    with pytest.raises(ValueError):
        make_manifest(
            sample,
            scientific_inputs={"missing": missing},
        )


def test_file_checksum_verification(sample):
    path = sample["scientific_file"]
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    record = file_record(
        path,
        expected_sha256=expected,
    )

    assert record["sha256"] == expected
    assert sha256_file(path) == expected

    with pytest.raises(
        ValueError,
        match="SHA256 mismatch",
    ):
        file_record(
            path,
            expected_sha256="0" * 64,
        )


def test_manifest_exclusive_publication(sample, tmp_path):
    manifest = make_manifest(sample)
    output = tmp_path / "manifest.json"

    write_manifest_exclusive(manifest, output)

    assert json.loads(output.read_text()) == manifest

    original = output.read_bytes()

    with pytest.raises(FileExistsError):
        write_manifest_exclusive(manifest, output)

    assert output.read_bytes() == original


def test_existing_symlink_not_overwritten(
    sample,
    tmp_path,
):
    manifest = make_manifest(sample)

    destination = tmp_path / "destination.json"
    destination.write_text("protected")

    link = tmp_path / "manifest.json"
    link.symlink_to(destination)

    with pytest.raises(FileExistsError):
        write_manifest_exclusive(manifest, link)

    assert destination.read_text() == "protected"
    assert link.is_symlink()


def test_scientific_input_checksum_recorded(sample):
    result = make_manifest(sample)

    expected = hashlib.sha256(
        sample["scientific_file"].read_bytes()
    ).hexdigest()

    assert result["scientific_inputs"][
        "events"
    ]["sha256"] == expected


def test_processing_script_checksum_recorded(sample):
    result = make_manifest(sample)

    expected = hashlib.sha256(
        sample["script_file"].read_bytes()
    ).hexdigest()

    assert result["processing_scripts"][
        "processor"
    ]["sha256"] == expected


def accepted_checksum(sample):
    path = sample["scientific_file"]
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_accepted_inventory_success(sample):
    expected = accepted_checksum(sample)

    manifest = make_manifest(
        sample,
        expected_scientific_sha256={
            "events": expected,
        },
    )

    assert manifest["verification"][
        "input_file_checksums"
    ] == "accepted_inventory_verified"

    assert manifest["scientific_inputs"][
        "events"
    ]["sha256"] == expected


def test_accepted_inventory_rejects_modified_input(sample):
    expected = accepted_checksum(sample)

    sample["scientific_file"].write_text(
        "sector,band\\nALTERED,h1012\\n"
    )

    with pytest.raises(
        ValueError,
        match="SHA256 mismatch",
    ):
        make_manifest(
            sample,
            expected_scientific_sha256={
                "events": expected,
            },
        )


def test_accepted_inventory_rejects_missing_entry(sample):
    with pytest.raises(
        ValueError,
        match="inventory",
    ):
        make_manifest(
            sample,
            expected_scientific_sha256={},
        )


def test_accepted_inventory_rejects_extra_entry(sample):
    expected = accepted_checksum(sample)

    with pytest.raises(
        ValueError,
        match="inventory",
    ):
        make_manifest(
            sample,
            expected_scientific_sha256={
                "events": expected,
                "unexpected": "0" * 64,
            },
        )


@pytest.mark.parametrize(
    "invalid",
    [
        "invalid",
        "0" * 63,
        "G" * 64,
        123,
        None,
    ],
)
def test_accepted_inventory_rejects_invalid_sha(
    sample,
    invalid,
):
    with pytest.raises(
        ValueError,
        match="Invalid accepted SHA256",
    ):
        make_manifest(
            sample,
            expected_scientific_sha256={
                "events": invalid,
            },
        )


def test_accepted_inventory_rejects_non_dictionary(sample):
    with pytest.raises(
        ValueError,
        match="dictionary",
    ):
        make_manifest(
            sample,
            expected_scientific_sha256=[
                "0" * 64,
            ],
        )


def test_without_inventory_preserves_previous_behavior(sample):
    manifest = make_manifest(sample)

    assert manifest["verification"][
        "input_file_checksums"
    ] == "computed"



def derived_fixture(sample):
    path = sample["scientific_file"].parent / "b2_guidance.csv"
    path.write_text("event_id,status\\n1,supported\\n")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, checksum


def test_derived_product_schema_v2(sample):
    path, checksum = derived_fixture(sample)

    manifest = make_manifest(
        sample,
        derived_products={"b2_guidance": path},
        expected_derived_sha256={"b2_guidance": checksum},
    )

    assert manifest["schema_version"] == "b2-provenance-2"
    assert manifest["derived_products"]["b2_guidance"][
        "sha256"
    ] == checksum
    assert manifest["verification"][
        "derived_product_checksums"
    ] == "accepted_inventory_verified"
    assert "b2_guidance" not in manifest["scientific_inputs"]


def test_derived_product_schema_v1_compatibility(sample):
    manifest = make_manifest(sample)

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert "derived_products" not in manifest
    assert "derived_product_checksums" not in (
        manifest["verification"]
    )


def test_derived_product_checksum_mismatch(sample):
    path, checksum = derived_fixture(sample)
    path.write_text("event_id,status\\n1,ALTERED\\n")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        make_manifest(
            sample,
            derived_products={"b2_guidance": path},
            expected_derived_sha256={
                "b2_guidance": checksum,
            },
        )


def test_derived_product_missing_checksum_inventory(sample):
    path, _ = derived_fixture(sample)

    with pytest.raises(ValueError, match="dictionary"):
        make_manifest(
            sample,
            derived_products={"b2_guidance": path},
        )


def test_derived_product_extra_checksum_entry(sample):
    path, checksum = derived_fixture(sample)

    with pytest.raises(ValueError, match="inventory"):
        make_manifest(
            sample,
            derived_products={"b2_guidance": path},
            expected_derived_sha256={
                "b2_guidance": checksum,
                "unexpected": "0" * 64,
            },
        )


@pytest.mark.parametrize(
    "invalid",
    ["invalid", "0" * 63, "G" * 64, 123, None],
)
def test_derived_product_invalid_checksum(sample, invalid):
    path, _ = derived_fixture(sample)

    with pytest.raises(
        ValueError,
        match="Invalid accepted derived SHA256",
    ):
        make_manifest(
            sample,
            derived_products={"b2_guidance": path},
            expected_derived_sha256={
                "b2_guidance": invalid,
            },
        )


def test_derived_product_logical_name_overlap(sample):
    path, checksum = derived_fixture(sample)

    with pytest.raises(ValueError, match="distinct logical names"):
        make_manifest(
            sample,
            derived_products={"events": path},
            expected_derived_sha256={"events": checksum},
        )


def test_derived_product_file_path_overlap(sample):
    path = sample["scientific_file"]
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="scientific input files"):
        make_manifest(
            sample,
            derived_products={"b2_guidance": path},
            expected_derived_sha256={
                "b2_guidance": checksum,
            },
        )


def test_derived_product_inventory_without_products(sample):
    with pytest.raises(
        ValueError,
        match="without derived products",
    ):
        make_manifest(
            sample,
            expected_derived_sha256={
                "b2_guidance": "0" * 64,
            },
        )


def test_derived_product_empty_inventory(sample):
    with pytest.raises(ValueError, match="cannot be empty"):
        make_manifest(
            sample,
            derived_products={},
            expected_derived_sha256={},
        )


def test_derived_product_non_dictionary(sample):
    with pytest.raises(ValueError, match="dictionary"):
        make_manifest(
            sample,
            derived_products=["b2_guidance"],
            expected_derived_sha256={},
        )


def test_derived_product_missing_file(sample):
    path = (
        sample["scientific_file"].parent
        / "missing_b2_guidance.csv"
    )

    with pytest.raises((FileNotFoundError, ValueError)):
        make_manifest(
            sample,
            derived_products={"b2_guidance": path},
            expected_derived_sha256={
                "b2_guidance": "0" * 64,
            },
        )
