"""Forecast agreement and deviation metrics (engineering specification §9).

Per completed interval, with saved forecast F and actual A (interval-average kW):

    deviation_kW        = A - F               (negative: actual below forecast)
    deviation_percent   = 100·(A - F)/A       (denominator: actual)
    abs_percentage_err  = 100·|F - A|/A
    accuracy_percent    = max(0, 100 - abs_percentage_err)

Row percentages are suppressed when A is at or below the low-output threshold.
Summaries use duration-weighted WAPE over paired intervals — never an average of
row percentages or a net daily energy difference, which lets over- and
under-prediction cancel.

"Accuracy %" is agreement with measured output, not a probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from .config import Installation, MeasurementBoundary

AGREEMENT_EXPLANATION = "Forecast agreement with measured output; not a probability."
FORECAST_SELECTION_RULE = "latest saved forecast issued at or before the interval start"


class ScoreStatus(str, Enum):
    SCORED = "scored"
    LOW_OUTPUT = "N/A — low output"
    AWAITING_DATA = "Awaiting actual data"
    INSUFFICIENT_DATA = "Insufficient data"
    NO_FORECAST = "No forecast issued before interval start"
    BOUNDARY_MISMATCH = "Measurement boundary mismatch"
    NOT_COMPLETED = "Interval not completed"


@dataclass(frozen=True)
class LowOutputThreshold:
    kw: float
    reference: str
    reference_capacity_kw: float
    measurement_floor_kw: float | None
    note: str


def low_output_threshold(installation: Installation, measurement_floor_kw: float | None = None) -> LowOutputThreshold:
    """Larger of 1 % of the reference capacity and the measurement uncertainty floor."""
    if installation.measurement_boundary is MeasurementBoundary.PV_DC_INPUT:
        reference, capacity_kw = "installed PV STC capacity", installation.stc_capacity_w / 1000.0
    else:
        reference, capacity_kw = "inverter AC rating", installation.inverter.ac_rated_w / 1000.0
    one_percent = 0.01 * capacity_kw
    if measurement_floor_kw is None:
        note = (
            f"provisional: 1 % of {reference} ({capacity_kw:g} kW); the measurement uncertainty floor is "
            "unknown and must be validated for this installation"
        )
        kw = one_percent
    else:
        kw = max(one_percent, measurement_floor_kw)
        note = f"larger of 1 % of {reference} ({one_percent:.3f} kW) and measurement floor ({measurement_floor_kw:g} kW)"
    return LowOutputThreshold(kw, reference, capacity_kw, measurement_floor_kw, note)


@dataclass(frozen=True)
class RowScore:
    status: ScoreStatus
    deviation_kw: float | None = None
    deviation_energy_kwh: float | None = None
    deviation_percent: float | None = None
    absolute_percentage_error: float | None = None
    accuracy_percent: float | None = None


def score_interval(
    forecast_kw: float | None, actual_kw: float | None, low_output_threshold_kw: float, duration_hours: float = 0.25
) -> RowScore:
    if forecast_kw is None:
        return RowScore(ScoreStatus.NO_FORECAST)
    if actual_kw is None:
        return RowScore(ScoreStatus.AWAITING_DATA)
    deviation = actual_kw - forecast_kw
    energy_dev = deviation * duration_hours
    if actual_kw <= low_output_threshold_kw:
        return RowScore(ScoreStatus.LOW_OUTPUT, deviation, energy_dev)
    ape = 100.0 * abs(forecast_kw - actual_kw) / actual_kw
    return RowScore(
        ScoreStatus.SCORED,
        deviation_kw=deviation,
        deviation_energy_kwh=energy_dev,
        deviation_percent=100.0 * deviation / actual_kw,
        absolute_percentage_error=ape,
        accuracy_percent=max(0.0, 100.0 - ape),
    )


@dataclass(frozen=True)
class PairedInterval:
    forecast_kw: float
    actual_kw: float
    duration_hours: float = 0.25


@dataclass(frozen=True)
class SummaryScore:
    label: str
    paired_intervals: int
    expected_intervals: int
    exclusions: Mapping[str, int]
    suppressed_row_percentages: int
    low_output_threshold_kw: float
    forecast_energy_kwh: float
    actual_energy_kwh: float
    wape_percent: float | None
    agreement_percent: float | None
    mae_kw: float | None
    rmse_kw: float | None
    mean_deviation_kw: float | None  # mean of (actual - forecast)
    net_energy_deviation_kwh: float  # sum of (actual - forecast) energy; can cancel, shown for context only
    selection_rule: str = FORECAST_SELECTION_RULE
    explanation: str = AGREEMENT_EXPLANATION

    @property
    def coverage_percent(self) -> float | None:
        if self.expected_intervals == 0:
            return None
        return 100.0 * self.paired_intervals / self.expected_intervals


def summarize(
    pairs: Sequence[PairedInterval],
    expected_intervals: int,
    low_output_threshold_kw: float,
    exclusions: Mapping[str, int] | None = None,
    label: str = "Today so far",
    aggregate_floor_kwh: float = 0.0,
) -> SummaryScore:
    n = len(pairs)
    abs_err_energy = sum(abs(p.forecast_kw - p.actual_kw) * p.duration_hours for p in pairs)
    actual_energy = sum(p.actual_kw * p.duration_hours for p in pairs)
    forecast_energy = sum(p.forecast_kw * p.duration_hours for p in pairs)
    wape = agreement = None
    if n and actual_energy > aggregate_floor_kwh:
        wape = 100.0 * abs_err_energy / actual_energy
        agreement = max(0.0, 100.0 - wape)
    deviations = [p.actual_kw - p.forecast_kw for p in pairs]
    return SummaryScore(
        label=label,
        paired_intervals=n,
        expected_intervals=expected_intervals,
        exclusions=dict(exclusions or {}),
        suppressed_row_percentages=sum(1 for p in pairs if p.actual_kw <= low_output_threshold_kw),
        low_output_threshold_kw=low_output_threshold_kw,
        forecast_energy_kwh=forecast_energy,
        actual_energy_kwh=actual_energy,
        wape_percent=wape,
        agreement_percent=agreement,
        mae_kw=sum(abs(d) for d in deviations) / n if n else None,
        rmse_kw=math.sqrt(sum(d * d for d in deviations) / n) if n else None,
        mean_deviation_kw=sum(deviations) / n if n else None,
        net_energy_deviation_kwh=actual_energy - forecast_energy,
    )
