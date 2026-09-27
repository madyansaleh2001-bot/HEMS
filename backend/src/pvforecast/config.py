"""Places, installations and equipment configuration.

Every value the physical model uses is traceable to its origin: entered by the
user, taken from a manufacturer datasheet, derived from other entries, or an
explicitly labelled assumption. Assumptions are reported as ``ModelNote``s so
they are never presented as measured or datasheet values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

REFRESH_STEP_MINUTES = 5

# Placeholder used only when a panel's temperature coefficient of Pmax is not
# supplied. It is always reported as an assumption, never as a datasheet value.
ASSUMED_GAMMA_PMP_PER_C = -0.0035

# Ground reflectance used when the user has not entered one (common default for
# generic ground; reported as an assumption).
ASSUMED_ALBEDO = 0.25

# Nominal inverter efficiency assumed for the *modelled* solar AC estimate when
# the inverter's value is not configured (pvlib PVWatts default).
ASSUMED_INVERTER_EFFICIENCY = 0.96


class ConfigError(ValueError):
    """Invalid or incomplete configuration."""


class MeasurementBoundary(str, Enum):
    """Where a power value is defined. Forecasts and actuals are only compared
    when they share the same boundary."""

    PV_DC_INPUT = "pv_dc_input"
    SOLAR_AC_OUTPUT = "solar_ac_output"


class Mounting(str, Enum):
    """Mounting style; selects the cell-temperature model parameters."""

    OPEN_RACK = "open_rack"
    CLOSE_ROOF_MOUNT = "close_roof_mount"
    INSULATED_BACK = "insulated_back"


class ModuleConstruction(str, Enum):
    GLASS_POLYMER = "glass_polymer"
    GLASS_GLASS = "glass_glass"


@dataclass(frozen=True)
class ModelNote:
    """An assumption, derivation or warning attached to a result."""

    kind: str  # "assumption" | "derived" | "warning" | "info"
    subject: str
    text: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.subject}: {self.text}"


def validate_refresh_minutes(minutes: Any) -> int:
    """Forecast refresh cadence must be a positive multiple of five minutes."""
    if isinstance(minutes, bool) or not isinstance(minutes, int):
        raise ConfigError(f"refresh interval must be an integer number of minutes, got {minutes!r}")
    if minutes <= 0 or minutes % REFRESH_STEP_MINUTES:
        raise ConfigError(
            f"refresh interval must be a positive multiple of {REFRESH_STEP_MINUTES} minutes, got {minutes}"
        )
    return minutes


def percent_per_c_to_per_c(value_percent_per_c: float) -> float:
    """Convert a datasheet coefficient such as -0.30 %/°C to -0.0030 /°C."""
    return float(value_percent_per_c) / 100.0


def resolve_timezone(latitude: float, longitude: float) -> str:
    """IANA time zone for the installation coordinates."""
    from timezonefinder import TimezoneFinder

    name = TimezoneFinder().timezone_at(lat=latitude, lng=longitude)
    if name is None:
        raise ConfigError(f"no time zone found for ({latitude}, {longitude}); enter it explicitly")
    return name


def _check_timezone(name: str) -> str:
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ConfigError(f"unknown IANA time zone {name!r}") from exc
    return name


@dataclass(frozen=True)
class Place:
    place_id: str
    name: str
    latitude: float
    longitude: float
    timezone: str
    elevation_m: float | None = None

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise ConfigError(f"latitude out of range: {self.latitude}")
        if not -180.0 <= self.longitude <= 180.0:
            raise ConfigError(f"longitude out of range: {self.longitude}")
        _check_timezone(self.timezone)


def make_place(
    place_id: str,
    name: str,
    latitude: float,
    longitude: float,
    timezone: str | None = None,
    elevation_m: float | None = None,
) -> Place:
    """Create a place, resolving its time zone from the coordinates when not given."""
    tz = timezone or resolve_timezone(latitude, longitude)
    return Place(place_id, name, float(latitude), float(longitude), tz, elevation_m)


@dataclass(frozen=True)
class PanelSpec:
    """One module type. ``source`` is "manual", "catalog" or "illustrative_example"."""

    source: str
    model_id: str | None = None
    rated_power_w: float | None = None  # STC
    efficiency: float | None = None  # fraction at STC, e.g. 0.215
    gamma_pmp_per_c: float | None = None  # 1/°C, e.g. -0.0030
    length_m: float | None = None
    width_m: float | None = None
    construction: ModuleConstruction | None = None
    catalog_record: str | None = None  # catalog id and revision when source == "catalog"

    @property
    def area_m2(self) -> float | None:
        if self.length_m and self.width_m:
            return self.length_m * self.width_m
        return None


@dataclass(frozen=True)
class ResolvedPanel:
    rated_power_w: float
    gamma_pmp_per_c: float
    construction: ModuleConstruction
    notes: tuple[ModelNote, ...]


def resolve_panel(panel: PanelSpec, subject: str = "panel") -> ResolvedPanel:
    """Fill in the values the model needs, recording every derivation or assumption.

    Rated STC power already reflects module efficiency, so efficiency is only used
    to derive a missing rating or to check consistency — never applied twice.
    """
    notes: list[ModelNote] = []
    area = panel.area_m2

    if panel.efficiency is not None and not 0.0 < panel.efficiency < 1.0:
        raise ConfigError(f"{subject}: efficiency must be a fraction between 0 and 1, got {panel.efficiency}")

    if panel.rated_power_w is not None:
        if panel.rated_power_w <= 0:
            raise ConfigError(f"{subject}: rated power must be positive")
        rated = float(panel.rated_power_w)
        if area is not None and panel.efficiency is not None:
            implied = rated / (area * 1000.0)
            if abs(implied - panel.efficiency) > 0.01:
                notes.append(ModelNote(
                    "warning", subject,
                    f"rated power {rated:.0f} W over {area:.3f} m² implies {implied:.1%} efficiency, "
                    f"but {panel.efficiency:.1%} was entered; check the datasheet values",
                ))
    elif area is not None and panel.efficiency is not None:
        rated = area * 1000.0 * panel.efficiency
        notes.append(ModelNote(
            "derived", subject,
            f"rated power {rated:.1f} W derived as area {area:.3f} m² × 1000 W/m² × efficiency {panel.efficiency:.1%}",
        ))
    else:
        raise ConfigError(f"{subject}: rated STC power is required (or module dimensions plus STC efficiency)")

    if panel.gamma_pmp_per_c is None:
        gamma = ASSUMED_GAMMA_PMP_PER_C
        notes.append(ModelNote(
            "assumption", subject,
            f"temperature coefficient of Pmax not supplied; assumed {gamma * 100:.2f} %/°C — replace with the datasheet value",
        ))
    else:
        gamma = float(panel.gamma_pmp_per_c)
        if not -0.02 <= gamma <= 0.0:
            raise ConfigError(
                f"{subject}: temperature coefficient {gamma} /°C is implausible; datasheet values in %/°C "
                "must be converted (e.g. -0.30 %/°C -> -0.0030 /°C)"
            )

    if panel.construction is None:
        construction = ModuleConstruction.GLASS_POLYMER
        notes.append(ModelNote(
            "assumption", subject, "module construction not supplied; glass/polymer backsheet assumed for temperature model",
        ))
    else:
        construction = panel.construction

    return ResolvedPanel(rated, gamma, construction, tuple(notes))


@dataclass(frozen=True)
class SubArray:
    """A group of identical modules sharing one fixed tilt and direction."""

    name: str
    panel: PanelSpec
    panel_count: int
    tilt_deg: float
    azimuth_deg: float  # 0 = north, 90 = east, 180 = south, 270 = west
    mounting: Mounting

    def __post_init__(self) -> None:
        if isinstance(self.panel_count, bool) or not isinstance(self.panel_count, int) or self.panel_count <= 0:
            raise ConfigError(f"{self.name}: panel count must be a positive integer")
        if not 0.0 <= self.tilt_deg <= 90.0:
            raise ConfigError(f"{self.name}: tilt must be between 0 and 90 degrees")
        if not 0.0 <= self.azimuth_deg < 360.0:
            raise ConfigError(f"{self.name}: azimuth must be in [0, 360) degrees (0 = north, 90 = east)")


@dataclass(frozen=True)
class InverterSpec:
    make: str
    model: str
    ac_rated_w: float
    battery_nominal_v: float | None = None
    nominal_efficiency: float | None = None
    # Applied as a DC harvest limit only when set; must be validated for the unit's revision first.
    pv_input_limit_w: float | None = None
    firmware_version: str | None = None
    notes: tuple[str, ...] = ()


def felicity_ivem4024_ii() -> InverterSpec:
    """The user's inverter. Ratings are user-confirmed; protocol details are not."""
    return InverterSpec(
        make="Felicity Solar",
        model="IVEM4024-II",
        ac_rated_w=4000.0,
        battery_nominal_v=24.0,
        notes=(
            "4 kW rated AC output and 24 V battery system confirmed by the user.",
            "The manufacturer product page lists 6,000 W maximum PV input; it is not applied as a "
            "harvest limit until validated for this unit's revision.",
            "RS232 protocol, firmware and PV-power register mapping are unverified; no telemetry is collected.",
        ),
    )


INVERTER_PRESETS = {"felicity_ivem4024_ii": felicity_ivem4024_ii}


@dataclass(frozen=True)
class Losses:
    """System losses in percent, applied multiplicatively to DC power (PVWatts method).

    Defaults are pvlib's PVWatts defaults except availability, which is 0 %:
    outages are flagged in scoring rather than averaged into the forecast.
    Fields named in ``user_set`` were entered by the user; others are assumptions.
    """

    soiling: float = 2.0
    shading: float = 3.0
    snow: float = 0.0
    mismatch: float = 2.0
    wiring: float = 2.0
    connections: float = 0.5
    lid: float = 1.5
    nameplate_rating: float = 1.0
    availability: float = 0.0
    user_set: frozenset[str] = field(default_factory=frozenset)

    NAMES = ("soiling", "shading", "snow", "mismatch", "wiring", "connections", "lid", "nameplate_rating", "availability")

    def __post_init__(self) -> None:
        for name in self.NAMES:
            value = getattr(self, name)
            if not 0.0 <= value < 100.0:
                raise ConfigError(f"loss {name} must be in [0, 100) percent, got {value}")

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.NAMES}

    def notes(self) -> tuple[ModelNote, ...]:
        assumed = [f"{n} {getattr(self, n):g} %" for n in self.NAMES if n not in self.user_set]
        if not assumed:
            return ()
        return (ModelNote("assumption", "losses", "PVWatts-style defaults, not site measurements: " + ", ".join(assumed)),)


@dataclass(frozen=True)
class Installation:
    installation_id: str
    place_id: str
    name: str
    config_version: int
    sub_arrays: tuple[SubArray, ...]
    inverter: InverterSpec
    losses: Losses = field(default_factory=Losses)
    measurement_boundary: MeasurementBoundary = MeasurementBoundary.PV_DC_INPUT
    refresh_minutes: int = 15
    albedo: float | None = None
    commissioning_date: date | None = None
    annual_degradation_percent: float | None = None
    battery_capacity_wh: float | None = None

    def __post_init__(self) -> None:
        if not self.sub_arrays:
            raise ConfigError("an installation needs at least one sub-array")
        validate_refresh_minutes(self.refresh_minutes)
        if self.albedo is not None and not 0.0 <= self.albedo <= 1.0:
            raise ConfigError("albedo must be between 0 and 1")
        if self.inverter.ac_rated_w <= 0:
            raise ConfigError("inverter AC rating must be positive")

    @property
    def stc_capacity_w(self) -> float:
        return sum(resolve_panel(a.panel).rated_power_w * a.panel_count for a in self.sub_arrays)


# --------------------------------------------------------------------------- JSON


def _panel_from_dict(d: Mapping[str, Any]) -> PanelSpec:
    if "gamma_pmp_per_c" in d and "gamma_percent_per_c" in d:
        raise ConfigError("give the temperature coefficient once: gamma_pmp_per_c or gamma_percent_per_c")
    gamma = d.get("gamma_pmp_per_c")
    if d.get("gamma_percent_per_c") is not None:
        gamma = percent_per_c_to_per_c(d["gamma_percent_per_c"])
    construction = d.get("construction")
    return PanelSpec(
        source=d.get("source", "manual"),
        model_id=d.get("model_id"),
        rated_power_w=d.get("rated_power_w"),
        efficiency=d.get("efficiency"),
        gamma_pmp_per_c=gamma,
        length_m=d.get("length_m"),
        width_m=d.get("width_m"),
        construction=ModuleConstruction(construction) if construction else None,
        catalog_record=d.get("catalog_record"),
    )


def _inverter_from_dict(d: Mapping[str, Any]) -> InverterSpec:
    if "preset" in d:
        try:
            base = INVERTER_PRESETS[d["preset"]]()
        except KeyError as exc:
            raise ConfigError(f"unknown inverter preset {d['preset']!r}") from exc
        overrides = {k: v for k, v in d.items() if k != "preset"}
        return InverterSpec(**{**base.__dict__, **overrides})
    return InverterSpec(**d)


def installation_from_dict(d: Mapping[str, Any], place_id: str) -> Installation:
    losses_in = dict(d.get("losses", {}))
    unknown = set(losses_in) - set(Losses.NAMES)
    if unknown:
        raise ConfigError(f"unknown loss names: {sorted(unknown)}")
    commissioning = d.get("commissioning_date")
    return Installation(
        installation_id=d["installation_id"],
        place_id=place_id,
        name=d.get("name", d["installation_id"]),
        config_version=int(d.get("config_version", 1)),
        sub_arrays=tuple(
            SubArray(
                name=a.get("name", f"array-{i + 1}"),
                panel=_panel_from_dict(a["panel"]),
                panel_count=a["panel_count"],
                tilt_deg=float(a["tilt_deg"]),
                azimuth_deg=float(a["azimuth_deg"]),
                mounting=Mounting(a.get("mounting", Mounting.CLOSE_ROOF_MOUNT.value)),
            )
            for i, a in enumerate(d["sub_arrays"])
        ),
        inverter=_inverter_from_dict(d["inverter"]),
        losses=Losses(**losses_in, user_set=frozenset(losses_in)),
        measurement_boundary=MeasurementBoundary(d.get("measurement_boundary", MeasurementBoundary.PV_DC_INPUT.value)),
        refresh_minutes=d.get("refresh_minutes", 15),
        albedo=d.get("albedo"),
        commissioning_date=date.fromisoformat(commissioning) if commissioning else None,
        annual_degradation_percent=d.get("annual_degradation_percent"),
        battery_capacity_wh=d.get("battery_capacity_wh"),
    )


def load_config(path: str | Path) -> tuple[Place, Installation]:
    """Load one place and one installation from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    p = data["place"]
    place = make_place(
        p["place_id"], p.get("name", p["place_id"]), p["latitude"], p["longitude"],
        timezone=p.get("timezone"), elevation_m=p.get("elevation_m"),
    )
    return place, installation_from_dict(data["installation"], place.place_id)
