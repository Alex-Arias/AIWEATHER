"""
High-level tropical cyclone verification workflow.

This module combines AIWeather tracking results with observed
best-track records and produces a unified verification result.
"""

from __future__ import annotations

from dataclasses import dataclass

from aiweather.tracking.records import TrackRecord
from aiweather.tracking.workflow import (
    TrackingWorkflowResult,
)

from .comparison import (
    TrackVerification,
    compare_forecast_to_best_track,
)


@dataclass(slots=True)
class VerificationWorkflowResult:
    """
    Unified verification result for available tropical cyclone tracks.
    """

    observations: list[TrackRecord]

    native: TrackVerification

    wuduan: TrackVerification | None = None

    vitart: TrackVerification | None = None

    @property
    def verifications(
        self,
    ) -> dict[str, TrackVerification]:
        """
        Return all available verification results.
        """
        result = {
            "native": self.native,
        }

        if self.wuduan is not None:
            result["wuduan"] = self.wuduan

        if self.vitart is not None:
            result["vitart"] = self.vitart

        return result


def verify_tracking_workflow(
    tracking_result: TrackingWorkflowResult,
    observation_records: list[TrackRecord],
    *,
    observation_name: str = "best_track",
) -> VerificationWorkflowResult:
    """
    Verify all available tracker outputs against observations.

    Parameters
    ----------
    tracking_result
        Result returned by ``evaluate_forecast_trackers``.

    observation_records
        Observed tropical cyclone best-track records.

    observation_name
        Label associated with the observations.

    Returns
    -------
    VerificationWorkflowResult
        Verification results for the native AIWeather track and
        available WuDuan and Vitart matches.
    """

    if not isinstance(
        tracking_result,
        TrackingWorkflowResult,
    ):
        raise TypeError(
            "tracking_result must be a "
            "TrackingWorkflowResult."
        )

    if not isinstance(
        observation_records,
        list,
    ):
        raise TypeError(
            "observation_records must be a list."
        )

    for record in observation_records:
        if not isinstance(
            record,
            TrackRecord,
        ):
            raise TypeError(
                "observation_records must contain "
                "TrackRecord objects."
            )

    native = compare_forecast_to_best_track(
        tracking_result.native_records,
        observation_records,
        forecast_name="native",
        observation_name=observation_name,
    )

    wuduan = None
    vitart = None

    evaluation = tracking_result.evaluation

    if evaluation is not None:

        if evaluation.wuduan_match is not None:
            wuduan = compare_forecast_to_best_track(
                evaluation.wuduan_match.records,
                observation_records,
                forecast_name="wuduan",
                observation_name=observation_name,
            )

        if evaluation.vitart_match is not None:
            vitart = compare_forecast_to_best_track(
                evaluation.vitart_match.records,
                observation_records,
                forecast_name="vitart",
                observation_name=observation_name,
            )

    return VerificationWorkflowResult(
        observations=observation_records,
        native=native,
        wuduan=wuduan,
        vitart=vitart,
    )