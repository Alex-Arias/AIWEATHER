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

    # ---------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------

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

        merged = cls._deep_merge(common, model)

        config = cls(values=merged)
        config.validate()

        return config

    # ---------------------------------------------------------
    # Internal utilities
    # ---------------------------------------------------------

    @staticmethod
    def _load_yaml(path: str | Path) -> dict[str, Any]:

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

    @staticmethod
    def _deep_merge(
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Recursively merge two dictionaries.
        """

        merged = dict(base)

        for key, value in override.items():

            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = Configuration._deep_merge(
                    merged[key],
                    value,
                )
            else:
                merged[key] = value

        return merged

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def validate(self) -> None:
        """
        Validate common configuration fields.
        """

        required = (
            "paths",
            "hardware",
        )

        missing = [
            key
            for key in required
            if key not in self.values
        ]

        if missing:
            raise ConfigurationError(
                "Missing required configuration keys: "
                + ", ".join(missing)
            )

    # ---------------------------------------------------------
    # Generic access
    # ---------------------------------------------------------

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:

        return self.values.get(key, default)

    def to_dict(self) -> dict[str, Any]:

        return dict(self.values)

    # ---------------------------------------------------------
    # Convenience properties
    # ---------------------------------------------------------

    @property
    def model_root(self) -> Path:
        return Path(self.values["paths"]["model_root"])

    @property
    def cache_root(self) -> Path:
        return Path(self.values["paths"]["cache_root"])

    @property
    def checkpoint_root(self) -> Path:
        return Path(self.values["paths"]["checkpoint_root"])

    @property
    def output_root(self) -> Path:
        return Path(self.values["paths"]["output_root"])

    @property
    def device(self) -> str:
        return self.values["hardware"]["device"]

    @property
    def precision(self) -> str:
        return self.values["hardware"]["precision"]

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:

        model = self.values.get("model_name", "undefined")

        dataset = self.values.get("dataset", "undefined")

        return (
            f"Configuration("
            f"model={model}, "
            f"dataset={dataset})"
        )