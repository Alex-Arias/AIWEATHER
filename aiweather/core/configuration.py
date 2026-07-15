"""
Configuration management for AIWeather.

This module loads, merges, and validates YAML configuration files
used throughout the AIWeather framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigurationError


@dataclass(slots=True)
class Configuration:
    """
    Represents a validated AIWeather configuration.

    Parameters
    ----------
    values
        Dictionary containing all configuration parameters.
    """

    values: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(
        cls,
        common_file: str | Path,
        model_file: str | Path,
    ) -> "Configuration":
        """
        Load and merge common and model-specific YAML files.
        """

        common = cls._load_yaml(common_file)
        model = cls._load_yaml(model_file)

        merged = {**common, **model}

        config = cls(values=merged)
        config.validate()

        return config

    @staticmethod
    def _load_yaml(path: str | Path) -> dict[str, Any]:
        """
        Load a YAML configuration file.
        """

        path = Path(path)

        if not path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {path}"
            )

        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)

        if data is None:
            return {}

        if not isinstance(data, dict):
            raise ConfigurationError(
                f"{path} must contain a YAML dictionary."
            )

        return data

    def validate(self) -> None:
        """
        Validate required configuration fields.
        """

        required = (
            "model_name",
            "dataset",
        )

        missing = [
            key for key in required
            if key not in self.values
        ]

        if missing:
            raise ConfigurationError(
                "Missing required configuration keys: "
                + ", ".join(missing)
            )

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Return a configuration value.
        """

        return self.values.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """
        Return a copy of the configuration dictionary.
        """

        return dict(self.values)

    def __repr__(self) -> str:
        return (
            f"Configuration("
            f"model={self.get('model_name')}, "
            f"dataset={self.get('dataset')})"
        )