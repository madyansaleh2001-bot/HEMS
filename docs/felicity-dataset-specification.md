# Dataset specification for AI Forecasting — Felicity installation

Prepared 27 September 2026. This is a data design and source assessment, not a tested inverter driver or a collected dataset. No inverter communication has been attempted. No additional sensors are required.

> Copied into the repository from the project handoff so the requirements travel with the code.

## 1. Equipment identity and what remains unverified

The user identifies the inverter as Felicity IVEM4024-II using RS232 and has confirmed a rated output of 4 kW and a 24 V battery system, correcting the earlier 5 kW description. The manufacturer's current product page agrees with these ratings and lists maximum PV input of 6,000 W. Maximum PV input and actual installed panel capacity are separate quantities; the user's panel-array capacity and firmware remain unknown. Validate any revision-specific input limits before configuring them. [Manufacturer product page](https://www.felicitysolar.com/product/ivem4024-ii/)

The manufacturer's manual describes a computer RS232 connection and an LCD PV-power reading. It does not, by itself, establish the complete command/register map and field definitions available from this particular unit. [Manufacturer manual downloads](https://www.felicitysolar.com/download/ivem4024-ii-user-guide-en/)

RS232 specifies the serial electrical interface. A collector still needs the exact application protocol, baud rate, framing, addressing, register/command definitions, scaling, signedness, and checksum rules. Do not assume that a driver for a different Felicity model or a generic Axpert command set is compatible. All proposed telemetry fields below are conditional on verified availability and meaning.

## 2. Recommended public datasets and code

### SolarDB — preferred research dataset

- [GitHub access library](https://github.com/PolasekT/SolarDB)
- [Dataset downloads and research description](https://cphoto.fit.vutbr.cz/solar/)
- [Dataset paper](https://cphoto.fit.vutbr.cz/solar/data/paper/polasek23solar.pdf)

SolarDB covers approximately one year and 16 plants. It provides DC/AC production on a standardized five-minute grid, including interpolated records, and hourly weather with archived forecasts. CSV and database downloads are available separately from the GitHub access code. Its forecast history makes it a useful candidate for realistic training with information available before generation occurs. The project restricts commercial dataset use without permission; the code's MIT license does not replace the dataset conditions.

Proposed use: aggregate production to the application's 15-minute intervals, preserve interpolation/quality information, select weather forecasts available by each issue time, and compare a physical baseline with a correction model. Evaluate transfer to the user's installation using local measurements. It is not an IVEM4024-II dataset or evidence of accuracy on this inverter.

### Solar Power Generation Data — simple Kaggle development dataset

- [Kaggle dataset by Ani Kannal](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)

The publisher describes two Indian plants, 34 days of records, and 15-minute generation data. Four files pair generation and weather-sensor data for each plant. Generation fields include DATE_TIME, PLANT_ID, SOURCE_KEY, DC_POWER, AC_POWER, DAILY_YIELD, and TOTAL_YIELD. The publisher's license label is "Data files © Original Authors."

Proposed use: exercise import mapping, plant/inverter grouping, quality checks, quarter-hour charts, and error calculations. Join plant-level weather by plant and timestamp, not by assuming the weather SOURCE_KEY is an inverter identifier. Validate power scaling, energy units, sampling meaning, and cumulative-counter resets before modeling. Measured weather at the target time is not an archived forecast available beforehand. Measured module-temperature inputs must not become mandatory features in the sensor-free application. This short dataset is not sufficient to demonstrate year-round performance at the user's site.

### Felicity-Inverter-Monitor — collection reference, not a dataset

- [GitHub repository](https://github.com/dj-nitehawk/Felicity-Inverter-Monitor)

Its author describes a Felicity monitor using RS232 Modbus and a USB serial connection on Windows/Linux. Treat it as implementation reference material. Compatibility with the user's exact IVEM4024-II firmware is unverified; it does not supply a ready historical training dataset. A second [ESPHome project](https://github.com/daoudeddy/fsolar_esphome) explicitly documents IVEM6048-II registers, which must not be copied into a 4024-II driver without validation.

No public historical dataset specifically verified for the user's IVEM4024-II was identified in this search. Public PV datasets are selected for their variables, provenance, climate, timing, and power-measurement boundary; matching the inverter brand alone does not establish suitability.

## 3. Define the learning target before collecting data

A hybrid inverter can supply the load from PV, battery discharge, and grid/generator input. Therefore total AC load/output power is not automatically PV generation. The earlier generic AC-output target needs an installation-specific measurement boundary.

Proposed primary target: **harvested PV power at the inverter's PV DC input**, if its telemetry exposes a verified reading. Store `measurement_boundary = pv_dc_input`. Preserve modeled solar AC output as a separately labeled estimate; only calculate AC accuracy if a matching solar-only AC measurement or validated energy-flow accounting is available.

Prefer a documented PV-input-power reading. If only PV voltage and current are available, derive `P_W = V_V × I_A` only after confirming both refer to the same PV input and synchronized samples. For multiple MPPT inputs, sum each channel's paired power. Battery-side solar-charging current multiplied by PV-side voltage is invalid. Battery-charging power is also not the whole PV harvest when PV simultaneously serves the load.

Keep available solar potential separate from harvested power: battery charge limits, a full battery, low load, or controller settings may curtail PV harvest. Record these conditions where possible. A counterfactual potential estimate cannot be validated directly against curtailed harvested power without accounting for restrictions. If neither a verified PV-power reading nor a valid same-boundary derivation exists, PV accuracy remains unavailable.

The AC inverter rating is not a universal cap on PV DC input, particularly when energy can also charge the battery. Use only validated limits appropriate to the selected boundary and device revision.

## 4. Canonical files and fields

CSV in UTF-8 is the simplest interchange format; XLSX can be accepted through a mapping screen. The following are proposed application field names, not confirmed Felicity register names. Missing unsupported values remain empty/null, never fabricated as zero. Keep raw source values and their conversion metadata.

### A. installations.csv — configuration records

| Field | Type / unit | Requirement |
|---|---|---|
| site_id, installation_id, config_version | Text | Required, stable identifiers and version |
| latitude, longitude, timezone | Degrees and IANA zone name | Required; actual installation location |
| inverter_make, inverter_model, firmware_version | Text | Model required; firmware recorded when accessible |
| inverter_ac_rated_w | W | 4000, confirmed by the user |
| pv_array_stc_w, panel_count, panel_model | W, integer, text | Required configuration or explicitly documented assumptions |
| tilt_deg, azimuth_deg | Degrees | Azimuth: 0 north, 90 east, 180 south, 270 west |
| battery_nominal_v, battery_capacity_wh | V, Wh | Nominal system voltage 24 V, confirmed by user; battery capacity remains to be supplied |
| measurement_boundary | Enum | For example pv_dc_input or separately verified solar_ac_output |
| effective_from_utc | ISO 8601 UTC | Preserves historical configuration changes |

Store detailed panel coefficients, mounting type, and loss assumptions in the main installation configuration. Do not treat serial baud settings as forecasting features.

### B. inverter_samples.csv — original time series

| Field | Unit / type | Purpose and availability |
|---|---|---|
| timestamp_utc, received_at_utc | ISO 8601 UTC | Sample time and ingestion time; required |
| site_id, installation_id, config_version | Text | Required association |
| pv_power_w | W | Preferred supervised target, conditional on verified PV-input meaning |
| pv_power_source | Enum/text | Register, per-MPPT V×I derivation, or imported production record |
| pv_voltage_v, pv_current_a | V, A | Useful validation/features if exposed; per-MPPT channel identifiers where needed |
| pv_energy_total_wh | Wh | Optional verified cumulative PV-energy counter; track resets |
| ac_load_power_w | W | Auxiliary load/output measurement, not automatically PV power |
| battery_voltage_v | V | Operational context if exposed |
| battery_current_a, battery_power_w | A, W | Normalize to positive charging and negative discharging; preserve original sign convention |
| battery_soc_pct, soc_source | %, text | Optional; distinguish BMS-reported SOC from an inverter estimate |
| grid_present, grid_input_power_w | Nullable boolean, W | Availability/input context if exposed |
| inverter_mode, charge_stage | Text/code | Solar/battery/bypass behavior and charging state if exposed |
| output_priority, charge_source_priority, charge_limit_a | Text/code, A | Existing operational settings, read only |
| inverter_temperature_c | °C | Internal inverter temperature if already available; not panel temperature |
| fault_code, warning_code | Text/code | Preserve source codes; decode only from verified definitions |
| curtailment_state, curtailment_reason | Unknown/active/inactive; text | An inference must be explicitly labeled inferred |
| quality_flag, protocol_profile_version | Text | Missing/stale/decode issues and the validated decoder version |

### C. weather_forecasts.csv — archived API inputs

Required identifiers/timing: `site_id`, `forecast_record_id`, `retrieved_at_utc`, `forecast_issued_at_utc` when supplied, `valid_start_utc`, `valid_end_utc`, `provider`, `model`, and `native_resolution_minutes`.

Candidate weather fields: `ghi_wm2`, `dni_wm2`, `dhi_wm2`, `air_temperature_c`, `wind_speed_ms`, `cloud_cover_pct`, `relative_humidity_pct`, `precipitation_mm`, and `surface_pressure_hpa`, subject to provider availability. Document instantaneous versus interval-mean or accumulated values. Store modeled `poa_irradiance_wm2_est` and `cell_temperature_c_est` as derived fields with model versions, not measured sensor inputs.

Preserve the provider's issue time separately from retrieval time. If issue time is unavailable, retain the original snapshot and retrieval timestamp; do not claim an upstream issue time. Training features can only use a snapshot available by the prediction issue time.

### D. pv_intervals_15min.csv — completed measurement intervals

Fields: `installation_id`, `config_version`, `interval_start_utc`, `interval_end_utc`, `measurement_boundary`, `actual_pv_mean_kw`, `actual_pv_energy_kwh`, `sample_count`, `coverage_seconds`, `quality_flag`, `operating_state_summary`, and `curtailment_state`.

Aggregate valid observations using time weighting. For complete 15-minute intervals, `actual_pv_mean_kw = actual_pv_energy_kwh / 0.25`. Do not obtain interval energy by simply summing watt readings or averaging irregular samples without duration weights. Verified cumulative PV-energy increments can provide interval energy when counter behavior is understood. Preserve nighttime zeros; mark gaps and partial coverage instead of filling missing observations with zero.

### E. forecast_predictions.csv — immutable predictions

Fields: `forecast_id`, `installation_id`, `config_version`, `issued_at_utc`, `interval_start_utc`, `interval_end_utc`, `measurement_boundary`, `forecast_pv_mean_kw`, `forecast_pv_energy_kwh`, `weather_snapshot_id`, `physical_model_version`, and `correction_model_version`.

Join actuals later by installation, boundary and interval. Keep forecast revisions separately. A derived evaluation view can include actual power, deviation, agreement percentage, lead time and quality/coverage flags. Use the same measurement boundary for each compared pair.

## 5. Collection and learning policy

Recommended initial acquisition target: one inverter sample approximately every 60 seconds, subject to the verified protocol and device limits. This is a proposed logging cadence, not a documented requirement or confirmed device capability. It is independent of the user-selected five-minute-multiple weather refresh and the 15-minute forecast display.

Proposed data path: existing inverter RS232 port → compatible communication adapter and local collector → stored telemetry → 15-minute aggregation → join with saved weather forecasts → physical model plus validated correction. A serial adapter/local computer is a communication bridge, not an additional weather sensor. No bridge is assumed to be installed already.

Use documented read-only queries; do not change charging/output settings or firmware to collect training data. Confirm the exact communication port and cable specification rather than treating an RJ45-shaped connector as Ethernet. Do not copy pinouts or serial settings from a different model.

Start with the physical forecast while local data accumulate. Public datasets help develop and benchmark the workflow; only later local validation supports a claim of accuracy at this site. A few weeks can support exploratory evaluation, while seasonal coverage requires much longer collection; deployment readiness depends on validation and weather/operational diversity, not elapsed days alone.

For every forecast example, include only operational history and weather forecasts available by its issue time. Later actual battery state or target-time weather cannot become an earlier forecast input. Keep all revisions of the same target interval in the same evaluation split. Separate normal-operation conversion modeling from curtailment/output-restriction modeling, and keep exclusions visible in accuracy reports.

## 6. Remaining integration evidence

The user has confirmed 4 kW output and a 24 V battery system for the stated IVEM4024-II. Before implementing an inverter driver, obtain firmware identification when available and the corresponding manufacturer RS232 protocol document or a verified export from the existing monitoring software. Then validate the available PV-power channel, units, signs, update rate, and boundary against the inverter display under known conditions. The public sources above do not by themselves establish register compatibility with this particular unit.
