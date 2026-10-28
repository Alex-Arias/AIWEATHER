"""Regression tests for experimental TC advisory integration."""

import pandas as pd
import pytest

from scripts.build_tc_swell_advisory import (
    build_tc_guidance,
    validate_forecast_times,
)


INIT = "20261005T120000"


def guidance_frames(
    *,
    families=3,
    position_supported=True,
    intensity_supported=True,
):
    """Construct synthetic position and intensity guidance."""

    times = pd.to_datetime([
        "2026-10-05 12:00:00",
        "2026-10-05 18:00:00",
    ])

    common = {
        "storm": ["rachel", "rachel"],
        "tracker": ["wuduan", "wuduan"],
        "valid_time": times,
        "lead_time_hours": [0, 6],
    }

    track = pd.DataFrame({
        **common,
        "n_model_families": [families] * 2,
        "guidance_supported": [position_supported] * 2,
    })

    intensity = pd.DataFrame({
        **common,
        "n_models": [families] * 2,
        "intensity_supported": [intensity_supported] * 2,
    })

    return track, intensity


def test_three_family_integration():
    track, intensity = guidance_frames()

    result = build_tc_guidance(track, intensity)

    assert len(result) == 2
    assert result["tc_jointly_supported"].all()
    assert (
        result["tc_availability_status"] == "supported"
    ).all()
    assert (
        result["tc_support_status"] == "supported"
    ).all()


@pytest.mark.parametrize(
    "families, expected",
    [
        (2, "limited"),
        (1, "insufficient"),
    ],
)
def test_reduced_family_availability(families, expected):
    track, intensity = guidance_frames(
        families=families,
        position_supported=False,
        intensity_supported=False,
    )

    result = build_tc_guidance(track, intensity)

    assert not result["tc_jointly_supported"].any()
    assert (
        result["tc_availability_status"] == expected
    ).all()
    assert (
        result["tc_support_status"] == expected
    ).all()


@pytest.mark.parametrize(
    "position_supported, intensity_supported",
    [
        (False, True),
        (True, False),
        (False, False),
    ],
)
def test_three_families_with_failed_support(
    position_supported,
    intensity_supported,
):
    track, intensity = guidance_frames(
        position_supported=position_supported,
        intensity_supported=intensity_supported,
    )

    result = build_tc_guidance(track, intensity)

    assert not result["tc_jointly_supported"].any()
    assert (
        result["tc_availability_status"] == "supported"
    ).all()
    assert (
        result["tc_support_status"] == "insufficient"
    ).all()


@pytest.mark.parametrize("families", [0, 4, 1.5, None])
def test_invalid_family_counts(families):
    track, intensity = guidance_frames(families=families)

    with pytest.raises(
        ValueError,
        match="Invalid independent model-family count",
    ):
        build_tc_guidance(track, intensity)


def test_family_count_mismatch():
    track, intensity = guidance_frames()

    intensity["n_models"] = 2

    with pytest.raises(
        ValueError,
        match="model-family counts differ",
    ):
        build_tc_guidance(track, intensity)


def test_two_family_false_support_rejected():
    track, intensity = guidance_frames(
        families=2,
        position_supported=True,
        intensity_supported=False,
    )

    with pytest.raises(
        ValueError,
        match="Upstream scientific support is inconsistent",
    ):
        build_tc_guidance(track, intensity)


def test_invalid_boolean_rejected():
    track, intensity = guidance_frames()

    track["guidance_supported"] = (
        track["guidance_supported"].astype(object)
    )
    track.loc[0, "guidance_supported"] = "not-a-bool"

    with pytest.raises(
        ValueError,
        match="Invalid support indicator",
    ):
        build_tc_guidance(track, intensity)


def test_string_false_remains_false():
    track, intensity = guidance_frames()

    track["guidance_supported"] = ["False", "True"]

    result = build_tc_guidance(track, intensity)

    assert result["tc_jointly_supported"].tolist() == [
        False,
        True,
    ]


def test_unmatched_timestamps_rejected():
    track, intensity = guidance_frames()

    intensity.loc[1, "valid_time"] = pd.Timestamp(
        "2026-10-06 00:00:00"
    )
    intensity.loc[1, "lead_time_hours"] = 12

    with pytest.raises(
        ValueError,
        match="timestamps differ",
    ):
        build_tc_guidance(track, intensity)


def test_valid_forecast_timestamps():
    track, intensity = guidance_frames()

    validate_forecast_times(track, INIT, "track")
    validate_forecast_times(intensity, INIT, "intensity")


def test_inconsistent_forecast_timestamp_rejected():
    track, _ = guidance_frames()

    track.loc[1, "valid_time"] = pd.Timestamp(
        "2026-10-05 19:00:00"
    )

    with pytest.raises(
        ValueError,
        match="valid_time does not match",
    ):
        validate_forecast_times(track, INIT, "track")


def test_negative_forecast_lead_rejected():
    track, _ = guidance_frames()

    track.loc[1, "lead_time_hours"] = -6

    with pytest.raises(
        ValueError,
        match="negative forecast lead times",
    ):
        validate_forecast_times(track, INIT, "track")


def test_empty_guidance_rejected():
    track, _ = guidance_frames()

    with pytest.raises(
        ValueError,
        match="empty guidance dataset",
    ):
        validate_forecast_times(
            track.iloc[0:0],
            INIT,
            "track",
        )
