import pytest

from aiweather.backends.earth2studio.data import (
    load_data_source,
)
from aiweather.registry.datasources import (
    get_data_source,
)


def test_graphcast_datasource_registry():
    config = get_data_source(
        "graphcast"
    )

    assert config["name"] == "gfs"
    assert config["class"].__name__ == "GFS"


def test_aifs2_datasource_registry():
    config = get_data_source(
        "aifs2"
    )

    assert config["name"] == "ifs"
    assert config["class"].__name__ == "IFS"


def test_unknown_model_datasource():
    with pytest.raises(
        ValueError,
        match="No datasource is configured",
    ):
        get_data_source(
            "unknown"
        )


def test_wrong_datasource_for_aifs2():
    with pytest.raises(
        ValueError,
        match="requires datasource 'ifs'",
    ):
        load_data_source(
            "aifs2",
            datasource="gfs",
        )


def test_wrong_datasource_for_graphcast():
    with pytest.raises(
        ValueError,
        match="requires datasource 'gfs'",
    ):
        load_data_source(
            "graphcast",
            datasource="ifs",
        )