from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import io
import os

import pytest

from aiweather.data.ibtracs import (
    IBTRACS_VERSION,
    download_ibtracs_dataset,
    ensure_ibtracs_dataset,
    ibtracs_cache_is_fresh,
    ibtracs_filename,
    ibtracs_url,
)


def test_ibtracs_filename():
    assert ibtracs_filename("EP") == (
        f"ibtracs.EP.list.{IBTRACS_VERSION}.csv"
    )

    assert ibtracs_filename("ep") == (
        f"ibtracs.EP.list.{IBTRACS_VERSION}.csv"
    )


def test_ibtracs_filename_invalid_basin():
    with pytest.raises(
        ValueError,
        match="unsupported",
    ):
        ibtracs_filename("XX")


def test_ibtracs_url():
    url = ibtracs_url("EP")

    assert url.endswith(
        f"/ibtracs.EP.list.{IBTRACS_VERSION}.csv"
    )

    assert "ncei.noaa.gov" in url


def test_ibtracs_cache_missing(tmp_path):
    path = (
        tmp_path
        / "missing.csv"
    )

    assert (
        ibtracs_cache_is_fresh(
            path
        )
        is False
    )


def test_ibtracs_cache_empty(tmp_path):
    path = (
        tmp_path
        / "empty.csv"
    )

    path.touch()

    assert (
        ibtracs_cache_is_fresh(
            path
        )
        is False
    )


def test_ibtracs_cache_fresh(tmp_path):
    path = (
        tmp_path
        / "ibtracs.csv"
    )

    path.write_text(
        "test\n"
    )

    now = datetime.now(
        timezone.utc
    )

    modified = (
        now
        - timedelta(hours=12)
    )

    os.utime(
        path,
        (
            modified.timestamp(),
            modified.timestamp(),
        ),
    )

    assert ibtracs_cache_is_fresh(
        path,
        max_age_hours=48.0,
        now=now,
    )


def test_ibtracs_cache_stale(tmp_path):
    path = (
        tmp_path
        / "ibtracs.csv"
    )

    path.write_text(
        "test\n"
    )

    now = datetime.now(
        timezone.utc
    )

    modified = (
        now
        - timedelta(hours=72)
    )

    os.utime(
        path,
        (
            modified.timestamp(),
            modified.timestamp(),
        ),
    )

    assert (
        ibtracs_cache_is_fresh(
            path,
            max_age_hours=48.0,
            now=now,
        )
        is False
    )


def test_ibtracs_cache_invalid_max_age(
    tmp_path,
):
    path = (
        tmp_path
        / "ibtracs.csv"
    )

    path.write_text(
        "test\n"
    )

    with pytest.raises(
        ValueError,
        match="max_age_hours",
    ):
        ibtracs_cache_is_fresh(
            path,
            max_age_hours=0.0,
        )


def test_download_ibtracs_dataset(
    monkeypatch,
    tmp_path,
):
    content = (
        b"SID,SEASON\n"
        b"test,2026\n"
    )

    calls = []

    def fake_urlopen(
        url,
        timeout,
    ):
        calls.append(
            (
                url,
                timeout,
            )
        )

        return io.BytesIO(
            content
        )

    monkeypatch.setattr(
        "aiweather.data.ibtracs.urlopen",
        fake_urlopen,
    )

    destination = (
        tmp_path
        / "ibtracs.csv"
    )

    result = download_ibtracs_dataset(
        "EP",
        destination,
    )

    assert result == destination

    assert destination.is_file()

    assert destination.read_bytes() == (
        content
    )

    assert len(calls) == 1

    assert calls[0][1] == 120


def test_ensure_ibtracs_dataset_reuses_fresh_cache(
    monkeypatch,
    tmp_path,
):
    destination = (
        tmp_path
        / ibtracs_filename("EP")
    )

    destination.write_text(
        "cached\n"
    )

    def fail_download(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "download should not occur"
        )

    monkeypatch.setattr(
        "aiweather.data.ibtracs."
        "download_ibtracs_dataset",
        fail_download,
    )

    result = ensure_ibtracs_dataset(
        basin="EP",
        cache_dir=tmp_path,
        max_age_hours=48.0,
    )

    assert result == destination


def test_ensure_ibtracs_dataset_downloads_missing(
    monkeypatch,
    tmp_path,
):
    calls = []

    def fake_download(
        basin,
        destination,
    ):
        calls.append(
            (
                basin,
                destination,
            )
        )

        destination.write_text(
            "downloaded\n"
        )

        return destination

    monkeypatch.setattr(
        "aiweather.data.ibtracs."
        "download_ibtracs_dataset",
        fake_download,
    )

    result = ensure_ibtracs_dataset(
        basin="EP",
        cache_dir=tmp_path,
    )

    assert result.is_file()

    assert calls == [
        (
            "EP",
            tmp_path
            / ibtracs_filename("EP"),
        )
    ]


def test_ensure_ibtracs_dataset_force_update(
    monkeypatch,
    tmp_path,
):
    destination = (
        tmp_path
        / ibtracs_filename("EP")
    )

    destination.write_text(
        "cached\n"
    )

    calls = []

    def fake_download(
        basin,
        destination,
    ):
        calls.append(
            basin
        )

        destination.write_text(
            "updated\n"
        )

        return destination

    monkeypatch.setattr(
        "aiweather.data.ibtracs."
        "download_ibtracs_dataset",
        fake_download,
    )

    result = ensure_ibtracs_dataset(
        basin="EP",
        cache_dir=tmp_path,
        force_update=True,
    )

    assert calls == [
        "EP",
    ]

    assert result.read_text() == (
        "updated\n"
    )


def test_ensure_ibtracs_dataset_invalid_force_update(
    tmp_path,
):
    with pytest.raises(
        TypeError,
        match="force_update",
    ):
        ensure_ibtracs_dataset(
            basin="EP",
            cache_dir=tmp_path,
            force_update="yes",
        )