import pytest

from pvforecast.accuracy import PairedInterval, ScoreStatus, low_output_threshold, score_interval, summarize
from pvforecast.config import MeasurementBoundary

from conftest import make_installation


def test_spec_example_4_80_forecast_vs_5_00_actual():
    s = score_interval(4.80, 5.00, low_output_threshold_kw=0.03)
    assert s.status is ScoreStatus.SCORED
    assert s.deviation_kw == pytest.approx(0.20)
    assert s.deviation_percent == pytest.approx(4.0)
    assert s.absolute_percentage_error == pytest.approx(4.0)
    assert s.accuracy_percent == pytest.approx(96.0)
    assert s.deviation_energy_kwh == pytest.approx(0.05)


def test_large_error_floors_agreement_but_keeps_the_error():
    s = score_interval(4.80, 2.00, low_output_threshold_kw=0.03)
    assert s.absolute_percentage_error == pytest.approx(140.0)
    assert s.accuracy_percent == 0.0
    assert s.deviation_kw == pytest.approx(-2.80)


def test_low_output_suppresses_percentages_but_keeps_kw_deviation():
    s = score_interval(0.10, 0.02, low_output_threshold_kw=0.03)
    assert s.status is ScoreStatus.LOW_OUTPUT
    assert s.accuracy_percent is None and s.deviation_percent is None
    assert s.deviation_kw == pytest.approx(-0.08)
    zero = score_interval(0.0, 0.0, low_output_threshold_kw=0.03)
    assert zero.status is ScoreStatus.LOW_OUTPUT and zero.accuracy_percent is None


def test_missing_values_are_never_scored():
    assert score_interval(1.0, None, 0.03).status is ScoreStatus.AWAITING_DATA
    assert score_interval(None, 1.0, 0.03).status is ScoreStatus.NO_FORECAST
    assert score_interval(1.0, None, 0.03).accuracy_percent is None


def test_summary_uses_wape_not_net_energy_or_row_average():
    pairs = [PairedInterval(2.0, 1.0), PairedInterval(1.0, 2.0)]
    s = summarize(pairs, expected_intervals=2, low_output_threshold_kw=0.03)
    assert s.net_energy_deviation_kwh == pytest.approx(0.0)  # cancels...
    assert s.wape_percent == pytest.approx(100 * 2 / 3)  # ...WAPE does not
    assert s.agreement_percent == pytest.approx(100 - 100 * 2 / 3)

    pairs = [PairedInterval(4.8, 5.0), PairedInterval(0.5, 1.0)]  # row agreements 96 % and 50 %
    s = summarize(pairs, expected_intervals=2, low_output_threshold_kw=0.03)
    assert s.agreement_percent == pytest.approx(100 - 100 * 0.7 / 6.0)
    assert s.agreement_percent != pytest.approx(73.0)


def test_summary_includes_low_output_intervals_and_counts_suppression():
    pairs = [PairedInterval(3.0, 3.0), PairedInterval(0.2, 0.0)]
    s = summarize(pairs, expected_intervals=4, low_output_threshold_kw=0.03, exclusions={"Awaiting actual data": 2})
    assert s.wape_percent == pytest.approx(100 * 0.2 / 3.0)
    assert s.suppressed_row_percentages == 1
    assert s.coverage_percent == pytest.approx(50.0)
    assert s.mae_kw == pytest.approx(0.1) and s.mean_deviation_kw == pytest.approx(-0.1)


def test_zero_aggregate_actual_is_not_applicable():
    s = summarize([PairedInterval(0.5, 0.0)], expected_intervals=1, low_output_threshold_kw=0.03)
    assert s.wape_percent is None and s.agreement_percent is None
    empty = summarize([], expected_intervals=10, low_output_threshold_kw=0.03)
    assert empty.agreement_percent is None and empty.mae_kw is None


def test_low_output_threshold_reference_capacity():
    dc = low_output_threshold(make_installation(panel_count=6, rated_w=500))
    assert dc.kw == pytest.approx(0.03) and "STC" in dc.reference and "provisional" in dc.note
    ac = low_output_threshold(make_installation(boundary=MeasurementBoundary.SOLAR_AC_OUTPUT))
    assert ac.kw == pytest.approx(0.04)
    floored = low_output_threshold(make_installation(), measurement_floor_kw=0.05)
    assert floored.kw == pytest.approx(0.05)
