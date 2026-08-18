from datetime import datetime

import pytest

from aiweather.output.manager import (
    OutputManager,
)


def test_build_verification_case_name():
    initialization_time = datetime(
        2026,
        7,
        24,
        0,
        0,
    )

    result = (
        OutputManager
        .build_verification_case_name(
            sid="2026204N08267",
            initialization_time=(
                initialization_time
            ),
        )
    )

    assert result == (
        "2026204N08267_"
        "20260724T000000"
    )


def test_build_verification_output_dir_without_create(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        OutputManager,
        "BASE_RESULTS",
        tmp_path,
    )

    initialization_time = datetime(
        2026,
        7,
        24,
        0,
        0,
    )

    result = (
        OutputManager
        .build_verification_output_dir(
            sid="2026204N08267",
            initialization_time=(
                initialization_time
            ),
            create=False,
        )
    )

    expected = (
        tmp_path
        / "verification"
        / (
            "2026204N08267_"
            "20260724T000000"
        )
    )

    assert result == expected
    assert result.exists() is False


def test_build_verification_output_dir_creates_directory(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        OutputManager,
        "BASE_RESULTS",
        tmp_path,
    )

    initialization_time = datetime(
        2026,
        7,
        24,
        0,
        0,
    )

    result = (
        OutputManager
        .build_verification_output_dir(
            sid="2026204N08267",
            initialization_time=(
                initialization_time
            ),
            create=True,
        )
    )

    assert result.is_dir()


def test_build_verification_case_name_empty_sid():
    initialization_time = datetime(
        2026,
        7,
        24,
        0,
        0,
    )

    with pytest.raises(
        ValueError,
        match="sid cannot be empty",
    ):
        (
            OutputManager
            .build_verification_case_name(
                sid="   ",
                initialization_time=(
                    initialization_time
                ),
            )
        )


def test_build_verification_case_name_invalid_sid_type():
    initialization_time = datetime(
        2026,
        7,
        24,
        0,
        0,
    )

    with pytest.raises(
        TypeError,
        match="sid must be a string",
    ):
        (
            OutputManager
            .build_verification_case_name(
                sid=123,
                initialization_time=(
                    initialization_time
                ),
            )
        )


def test_build_verification_case_name_invalid_initialization_time():
    with pytest.raises(
        TypeError,
        match="initialization_time",
    ):
        (
            OutputManager
            .build_verification_case_name(
                sid="2026204N08267",
                initialization_time=(
                    "2026-07-24T00:00:00"
                ),
            )
        )


def test_build_verification_output_dir_invalid_create():
    initialization_time = datetime(
        2026,
        7,
        24,
        0,
        0,
    )

    with pytest.raises(
        TypeError,
        match="create must be a boolean",
    ):
        (
            OutputManager
            .build_verification_output_dir(
                sid="2026204N08267",
                initialization_time=(
                    initialization_time
                ),
                create="yes",
            )
        )