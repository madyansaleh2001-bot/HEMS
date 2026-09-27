"""Physical PV baseline (pvlib).

Pipeline per fixed sub-array, on 1-minute steps:
  sun position -> Perez transposition to the panel plane -> physical IAM on the
  beam component -> SAPM cell temperature -> PVWatts DC -> PVWatts system losses.

The SAPM temperature model is used because pvlib ships documented parameter
sets per mounting type and the model is defined with wind speed at 10 m, which
is the height weather APIs report. Cell temperature and plane-of-array
irradiance are *estimates* — there are no on-site sensors.

Outputs are kept at separate measurement boundaries:
  * ``pv_dc_w``        unconstrained PV potential at the inverter's PV DC input
                       (battery-full / low-load curtailment is not modelled);
  * ``solar_ac_w_est`` a modelled solar-only AC estimate, clipped at the
                       inverter AC rating. The DC value is never clipped to the
                       AC rating.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd
import pvlib

from .config import (
    ASSUMED_ALBEDO,
    ASSUMED_INVERTER_EFFICIENCY,
    Installation,
    ModelNote,
    ModuleConstruction,
    Mounting,
    Place,
    resolve_panel,
)
from .weather import FINE_STEP, WeatherSnapshot, to_fine_resolution

PHYSICAL_MODEL_VERSION = "physics-1.0 (pvlib Perez POA, physical IAM, SAPM cell temperature, PVWatts DC + losses)"

_SAPM = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]


def sapm_parameters(mounting: Mounting, construction: ModuleConstruction) -> tuple[str, dict[str, float]]:
    if mounting is Mounting.OPEN_RACK:
        key = "open_rack_glass_glass" if construction is ModuleConstruction.GLASS_GLASS else "open_rack_glass_polymer"
    elif mounting is Mounting.CLOSE_ROOF_MOUNT:
        key = "close_mount_glass_glass"  # the only close-mount set pvlib provides
    else:
        key = "insulated_back_glass_polymer"
    return key, _SAPM[key]


@dataclass(frozen=True)
class PowerSeries:
    fine: pd.DataFrame  # 1-minute bins (UTC bin starts); powers in W
    notes: tuple[ModelNote, ...]


def _age_loss_percent(installation: Installation, on: date) -> tuple[float, ModelNote | None]:
    if installation.commissioning_date is None or installation.annual_degradation_percent is None:
        return 0.0, ModelNote(
            "assumption", "degradation", "no module degradation applied (commissioning date and annual rate not both supplied)"
        )
    years = max(0.0, (on - installation.commissioning_date).days / 365.25)
    return years * installation.annual_degradation_percent, ModelNote(
        "derived", "degradation",
        f"{years:.1f} years × {installation.annual_degradation_percent:g} %/year user-entered degradation",
    )


def model_power(
    place: Place,
    installation: Installation,
    snapshot: WeatherSnapshot,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    daylight_spans: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
) -> PowerSeries:
    """Modelled power on 1-minute bins over [start, end).

    Power is zero outside ``daylight_spans`` (actual sunrise to sunset). Inside
    them, missing weather yields NaN, never zero.
    """
    wx, weather_notes = to_fine_resolution(place, snapshot, start_utc, end_utc)
    notes: list[ModelNote] = list(weather_notes)
    for required in ("air_temperature_c", "wind_speed_ms"):
        if required not in wx.columns:
            raise ValueError(f"weather snapshot lacks {required}, required for the cell-temperature model")

    albedo = installation.albedo
    if albedo is None:
        albedo = ASSUMED_ALBEDO
        notes.append(ModelNote("assumption", "albedo", f"ground reflectance {albedo} assumed"))

    total_dc = np.zeros(len(wx))
    poa_weighted = np.zeros(len(wx))
    tcell_weighted = np.zeros(len(wx))
    capacity = 0.0
    for array in installation.sub_arrays:
        panel = resolve_panel(array.panel, subject=array.name)
        notes.extend(panel.notes)
        pdc0 = panel.rated_power_w * array.panel_count
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=array.tilt_deg,
            surface_azimuth=array.azimuth_deg,
            solar_zenith=wx["apparent_zenith"],
            solar_azimuth=wx["azimuth"],
            dni=wx["dni_wm2"],
            ghi=wx["ghi_wm2"],
            dhi=wx["dhi_wm2"],
            dni_extra=wx["dni_extra"],
            airmass=wx["airmass_relative"],
            albedo=albedo,
            model="perez",
        )
        # Perez divides by DHI; with zero diffuse (e.g. a zero-irradiance forecast in daylight)
        # the sky-diffuse term is zero, not missing.
        no_diffuse = (wx["dhi_wm2"] <= 0).to_numpy()
        if no_diffuse.any():
            poa.loc[no_diffuse, "poa_sky_diffuse"] = 0.0
            poa["poa_diffuse"] = poa["poa_sky_diffuse"] + poa["poa_ground_diffuse"]
            poa["poa_global"] = poa["poa_direct"] + poa["poa_diffuse"]
        aoi =pvlib.irradiance.aoi(array.tilt_deg, array.azimuth_deg, wx["apparent_zenith"], wx["azimuth"])
        effective = poa["poa_direct"] * pvlib.iam.physical(aoi) + poa["poa_diffuse"]
        key, params = sapm_parameters(array.mounting, panel.construction)
        tcell = pvlib.temperature.sapm_cell(poa["poa_global"], wx["air_temperature_c"], wx["wind_speed_ms"], **params)
        pdc = pvlib.pvsystem.pvwatts_dc(effective, tcell, pdc0=pdc0, gamma_pdc=panel.gamma_pmp_per_c)
        notes.append(ModelNote("info", array.name, f"cell temperature: SAPM '{key}' parameters (estimate, no sensor)"))

        total_dc = total_dc + pdc.to_numpy()
        poa_weighted = poa_weighted + poa["poa_global"].to_numpy() * pdc0
        tcell_weighted = tcell_weighted + tcell.to_numpy() * pdc0
        capacity += pdc0

    age_pct, age_note = _age_loss_percent(installation, start_utc.date())
    notes.append(age_note)
    notes.extend(installation.losses.notes())
    loss_pct = float(pvlib.pvsystem.pvwatts_losses(**installation.losses.as_dict(), age=age_pct))
    pv_dc = total_dc * (1.0 - loss_pct / 100.0)
    if installation.inverter.pv_input_limit_w is not None:
        pv_dc = np.minimum(pv_dc, installation.inverter.pv_input_limit_w)
        notes.append(ModelNote("info", "inverter", f"PV harvest limited to configured {installation.inverter.pv_input_limit_w:g} W input"))

    eta = installation.inverter.nominal_efficiency
    if eta is None:
        eta = ASSUMED_INVERTER_EFFICIENCY
        notes.append(ModelNote(
            "assumption", "solar AC estimate",
            f"inverter efficiency {eta:.0%} assumed; the solar AC column is a modelled estimate only",
        ))
    ac_rated = installation.inverter.ac_rated_w
    solar_ac = pvlib.inverter.pvwatts(pd.Series(pv_dc), pdc0=ac_rated / eta, eta_inv_nom=eta).to_numpy()
    solar_ac = np.minimum(solar_ac, ac_rated)

    mid = wx.index + FINE_STEP / 2
    daylight = np.zeros(len(wx), dtype=bool)
    for span_start, span_end in daylight_spans:
        daylight |= (mid >= span_start) & (mid <= span_end)

    out = pd.DataFrame(index=wx.index)
    out["pv_dc_w"] = np.where(daylight, pv_dc, 0.0)
    out["solar_ac_w_est"] = np.where(daylight, solar_ac, 0.0)
    out["poa_global_wm2_est"] = np.where(daylight, poa_weighted / capacity, 0.0)
    out["cell_temp_c_est"] = tcell_weighted / capacity
    out["daylight"] = daylight
    for column in ("ghi_wm2", "air_temperature_c", "wind_speed_ms", "cloud_cover_pct", "precipitation_mm"):
        if column in wx.columns:
            out[column] = wx[column].to_numpy()
    notes.append(ModelNote("info", "losses", f"total DC system loss {loss_pct:.2f} %"))
    return PowerSeries(out, _dedupe(notes))


def _dedupe(notes: Sequence[ModelNote]) -> tuple[ModelNote, ...]:
    return tuple(dict.fromkeys(notes))


def integrate_energy_kwh(fine: pd.DataFrame, column: str, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    """Energy over [start, end) from 1-minute bins, weighting partial bins by overlap.

    Returns None if any overlapping bin has no value (missing weather).
    """
    bin_start = fine.index
    bin_end = bin_start + FINE_STEP
    lo = np.maximum(bin_start.as_unit("ns").asi8, pd.Timestamp(start).value)
    hi = np.minimum(bin_end.as_unit("ns").asi8, pd.Timestamp(end).value)
    overlap_h = np.clip(hi - lo, 0, None) / 3.6e12
    used = overlap_h > 0
    span_h = (pd.Timestamp(end) - pd.Timestamp(start)) / pd.Timedelta(hours=1)
    if abs(overlap_h.sum() - span_h) > 1e-9:
        raise ValueError(f"modelled series does not cover {start} – {end}")
    values = fine[column].to_numpy(dtype="float64")
    if np.isnan(values[used]).any():
        return None
    return float(np.sum(values[used] * overlap_h[used]) / 1000.0)
