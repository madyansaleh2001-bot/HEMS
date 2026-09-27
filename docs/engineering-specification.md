# AI Forecasting: PV power generation forecasting and simulation

Engineering specification — created 26 September 2026; updated 27 September 2026

> Copied into the repository from the project handoff so the requirements travel with the code.
> Confirmed user requirements and engineering recommendations are distinguished in the text.

Status: revised proposed design incorporating the user's clarifications. Confirmed: today's full sunrise-to-sunset table with quarter-hour rounding, interval-average power with interval energy, actual generation beside forecasts for completed intervals, forecast accuracy percentages, multiple editable GPS-based places/installations, future dataset uploads, and a proposed self-learning workflow. Additional sensors are unavailable and are excluded from the design. The separate gadget summarizes tomorrow. No dataset is available yet. No application, live installation, weather connection, or trained AI model has been configured. Initial coordinates and preferred first implementation deliverable remain to be supplied.

## 1. Purpose and operating modes

Predict the output of each saved fixed PV installation from its specifications, location, and forecast weather. Display a sunrise-to-sunset forecast with 15-minute rows and an independently updated next-day summary. The earlier six-hour example is superseded by the daylight-period requirement; row count depends on daylight duration.

- **Forecast:** use the saved installation and the latest available weather forecast.
- **Simulation:** compare changes in weather, panel count, orientation, or equipment without overwriting the saved installation. Clearly mark scenario results.
- **Dataset evaluation:** compare predictions against measured generation once suitable historical data are supplied.

The first installation uses the user's Felicity IVEM4024-II hybrid/off-grid inverter, with 4 kW rated output and a 24 V battery system confirmed by the user, communicating over RS232. For this installation, prefer verified PV DC input power as the measured solar-generation target, subject to protocol validation. Total AC output/load power can include battery discharge or grid/generator contribution and must not automatically be labeled actual PV generation. Show modeled solar AC output separately and calculate its accuracy only if matching solar-only measurements or validated energy-flow accounting exist. Every forecast, actual reading, metric and learned model must carry an explicit measurement boundary. This installation-specific choice supersedes the earlier generic AC-output default.

See [Felicity dataset and collection specification](felicity-dataset-specification.md) for public GitHub/Kaggle sources, telemetry fields, the confirmed equipment rating, and outstanding protocol checks. The RS232 interface alone does not establish register compatibility or data availability. No additional sensors are required.

## 2. Installation configuration

| Input | Behavior |
|---|---|
| Places and GPS location | Add and name multiple places; set the first and later locations using browser location sharing or manually entered GPS coordinates; review and edit coordinates for each place |
| Installations | Add and name installations under a place; select the active installation; preserve separate equipment, orientation, settings, and history for each |
| Time zone | Resolve from installation coordinates; display dates and times in that zone |
| Panel entry | Manual specifications or selection of an exact catalog model and power variant |
| Panel count | Positive integer; support separate fixed sub-arrays when orientations differ |
| Rated module power | Watts at standard test conditions (W / Wp) |
| Module efficiency | Percentage, with dimensions/area for consistency checks |
| Temperature coefficient of maximum power | Store in 1/°C; convert a datasheet value such as -0.30 %/°C to -0.003 /°C |
| Module dimensions | Length and width, with units; useful for area and efficiency validation |
| Mounting | Roof or open rack; clearance/ventilation assumptions for temperature modeling |
| Tilt | Degrees from horizontal; fixed during forecasting |
| Direction / azimuth | Interface convention: 0° north, 90° east, 180° south, 270° west |
| Inverter | AC rated capacity and efficiency/model; required to estimate clipping and usable AC output |
| Losses | Editable wiring, mismatch, soiling, shading, and availability assumptions, with defaults explicitly identified |
| Commissioning date | Optional; enables a documented degradation assumption |
| Forecast refresh | Positive multiples of five minutes: 5, 10, 15, 20, and so on |

For manual entry, missing advanced parameters must be identified as assumptions, not silently presented as measured specifications. The rated wattage already reflects module efficiency: do not multiply rated power by efficiency again. If rated power is missing but area and STC efficiency are available, estimate it as area × 1,000 W/m² × efficiency and label it as derived.

Use the actual forecast timestamp to determine season and month. A manually selected month belongs to historical or hypothetical simulation; it must not override the date of a live forecast.

Save each place with a stable identifier, name, coordinates, and time zone, and each installation with its own stable identifier and configuration version. Changing the active installation must switch all weather, sunrise/sunset, forecast, dataset, and summary views together. Editing coordinates creates a new configuration version and recalculates location-dependent values; existing forecasts and uploaded measurements retain their original site/configuration association. Treat a physically separate installation as an additional installation rather than reassigning another installation's history. Each saved installation can have its own refresh schedule in five-minute increments. Do not infer the first site's coordinates from the user's computer location or time zone.

## 3. Panel catalog and installation guidance

Provide brand → series → exact model / power variant selection. Proposed initial catalog families include [JinkoSolar Tiger Neo](https://www.jinkosolar.com/en/site/tigerneo), [LONGi Hi-MO X10](https://wf-eu.longi.com/hi-mo-X10), and [Trina Solar Vertex S+](https://vertexsplus.trinasolar.com/). These are candidate families, not a verified popularity ranking.

Populate each model from its manufacturer datasheet. Preserve the source URL, revision, retrieval date, exact model identifier, rated power, efficiency, temperature coefficient, dimensions, and available electrical parameters. Keep catalog records versioned so a later update cannot silently change an existing installation. Where appropriate, compare with pvlib's [SAM/CEC module database support](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.retrieve_sam.html); database coverage of a specific current model must be checked.

Offer a suggested fixed tilt and direction that maximize estimated annual energy, subject to roof constraints. PVGIS supports optimization of slope and orientation for annual production. Present the recommendation separately from the user's actual installation, which remains fixed unless edited. GPS cannot identify nearby trees, buildings, or roof obstacles; these require user input or a site survey. [PVGIS documentation](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-tools/grid-connected-pv_en)

## 4. Weather acquisition

Proposed initial provider: Open-Meteo, subject to location coverage and applicable service limits. Its API exposes solar radiation, temperature, wind, humidity, precipitation, cloud information, and daily sunrise/sunset. Native 15-minute data are region-dependent; other regions use hourly interpolation. Weather model updates are less frequent than a five-minute application refresh. Show data age and resolution honestly. [Open-Meteo forecast documentation](https://open-meteo.com/en/docs)

Collect relevant available weather features rather than claiming every atmospheric variable is known:

- Core modeling inputs: global/direct/diffuse irradiance, air temperature, wind speed.
- Additional candidate features: cloud layers, humidity, dew point, pressure, precipitation, snow, visibility, and wind direction.
- Optional extensions: aerosol/dust information and ground albedo, where a suitable source exists.

Convert provider units and azimuth conventions at the API boundary. Use one irradiance workflow: either transpose horizontal radiation onto the panel plane or use provider-supplied tilted radiation; do not transpose twice. Do not apply a second arbitrary cloud/rain penalty when those effects are already represented in forecast irradiance.

Store retrieval time separately from the upstream forecast issue time, when available. Preserve the provider/model identity, original temporal resolution, missing-data flags, and interpolation method. A retrieval timestamp is not proof that a new model forecast has been issued.

## 5. Forecast model and the AI component

Start with a reproducible physical baseline using pvlib:

1. Compute sun position from coordinates and time.
2. Calculate radiation reaching each fixed panel plane.
3. Estimate cell temperature from irradiance, air temperature, wind, and mounting assumptions.
4. Calculate DC power, then apply documented losses and an inverter model.
5. Integrate over each requested interval to obtain average power and energy at the selected measurement boundary; keep PV DC and modeled solar AC results separate.

A simplified baseline equation is:

`Pdc_W = N × Pmodule_STC_W × (Geffective_Wm2 / 1000) × [1 + gamma_per_C × (Tcell_C - 25)]`

The model uses cell temperature, which differs from ambient temperature. The detailed implementation should use the selected pvlib model and its documented parameter conventions. [PVWatts DC model](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html), [Faiman temperature model](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.faiman.html)

Once measured generation is available, evaluate a supervised model that predicts corrections to the physical baseline. Candidate algorithms can be compared using time-ordered validation rather than selected in advance as the presumed winner. Include lead time, weather, solar geometry, and physical-model output as features; include past measured power only when it will also be available during operation.

Use actual generation as the training target. Physics-only forecasts remain available before training. Synthetic model outputs are useful for testing but do not establish real-world forecasting accuracy.

### 5.1 Recommended additional variables

The following priorities are design recommendations to test on each installation, not a promise that every variable will improve accuracy. Keep a variable only if it is available during operation and improves time-ordered validation. The physical calculation follows the irradiance → temperature → electrical-output sequence described by [Sandia PVPMC](https://pvpmc.sandia.gov/modeling-guide/).

Confirmed equipment constraint: no additional weather, irradiance, panel-temperature, or soiling sensors are available. Use weather APIs, installation configuration, user maintenance/event entries, production-log uploads, and existing inverter/monitoring telemetry only when that telemetry is accessible. Do not require new measurement hardware or include a sensor-upgrade step in the current scope. Label modeled quantities as estimates; the availability of an inverter API or production export remains to be established.

| Priority | Variable group | Proposed source and use |
|---|---|---|
| Essential for learning and scoring | Actual interval energy/power at the selected PV measurement boundary; timestamps and measurement quality | Existing inverter/monitoring exports or production-log upload; a live connection only if supported by existing equipment. Prefer verified PV DC readings for the specified hybrid inverter. Supplies the learning target and comparison data. Accurate time alignment is required. Physical forecasts can run before these observations exist. |
| Essential | Forecast irradiance, cloud layers, temperature and wind; issue time, provider/model and lead time | Weather API plus archived original responses. Distinguishes weather situations and short-lead forecasts from forecasts issued much earlier. |
| Essential | Solar elevation/azimuth, season, panel tilt/direction and configuration version | Calculated from GPS/time plus setup. Encodes geometry and configuration changes. |
| High, if live data exist | Measured power over the previous 15, 30 and 60 minutes; recent measured-minus-predicted residuals | Uses only completed measurements received before forecast issue time. Captures recent site behavior; unavailable future measurements cannot be input features. |
| High when already accessible | Inverter operating status, fault codes, AC rating, power limit, temperature, and DC voltage/current | Existing inverter telemetry or logs; no added sensors. Helps identify clipping, thermal limits, faults, and unavailable equipment. Missing telemetry is unknown, not evidence of healthy operation. |
| High | Estimated cell temperature and mounting ventilation | Calculate from weather-API air temperature, wind, estimated incident irradiance, and mounting assumptions. Label as modeled; there is no measured panel-temperature input. Ambient temperature is not interchangeable with cell temperature. |
| High | Local shading profile by solar direction/elevation, obstacles, row spacing | Site setup and observations. Describes recurrent building/tree/row shadows; GPS alone cannot observe these. |
| High | Cleaning date, estimated soiling, rainfall since cleaning, maintenance events | User event log, available weather data, and conservative learned loss adjustments. There is no measured soiling input. A cleaning event can change the site's learned loss level; residual errors alone cannot establish that dust caused a loss. |
| Core modeled quantity | Estimated irradiance in the plane of the array | Transform API irradiance using the saved panel geometry, or use a documented provider tilted-irradiance estimate. This is not an on-site irradiance measurement. |
| Conditional | Grid availability, export limits, curtailment command; battery state of charge, charging limits and load | Relevant when the installation controller restricts PV output. Treat these as operational conditions, not direct changes to the panels' solar conversion. Household load alone is not a physical irradiance input. |
| Conditional | Snow cover estimate, ground reflectance assumption, estimated rear irradiance and bifaciality | Use weather data and user setup for applicable climates and bifacial installations; only model rear-side contribution when installation geometry and assumptions support it. |
| Longer term | Module age, repairs and replacements; string-level current imbalance if already logged | Maintenance records and existing telemetry. Track gradual degradation separately from abrupt equipment/configuration changes. |

The temperature-model inputs are documented by [pvlib](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.faiman.html). Soiling, shading, snow, mismatch, wiring, age, and availability are recognized loss categories in its [PVWatts losses model](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_losses.html). Inverter clipping and instrument condition can obscure performance problems, so diagnosis should not rely on output power alone. [Sandia discussion](https://www.sandia.gov/research/publications/details/masking-of-photovoltaic-system-performance-problems-by-inverter-clipping-an-2021-07-01/)

The workflow uses setup information, archived forecasts, and generation logs without additional sensors. Periodic CSV/XLSX uploads permit learning after import; automatic adaptation to current measurements requires an accessible, timely feed from existing inverter/monitoring equipment. Without actual production observations, continue physical forecasts and show actuals/accuracy as unavailable; do not present modeled production as a learning target. Without measured local irradiance and module temperature, weather errors and equipment/soiling losses can be difficult to separate, so report diagnostic causes as hypotheses rather than confirmed measurements.

### 5.2 Proposed self-learning workflow

First candidate: a pvlib physical baseline plus a gradient-boosted-tree residual model. The learning target is `actual_PV_kW - baseline_PV_kW` at the same selected measurement boundary, normalized by the corresponding installation capacity where appropriate. For the specified hybrid inverter, use verified harvested PV DC readings rather than mixed-source AC load power. Evaluate a simple recent-bias correction as a competing baseline before using a more complex model. A candidate implementation is scikit-learn's [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html); this is an engineering choice to validate, not a claim that it is the best algorithm for unseen data.

1. Save immutable forecast inputs, the physical output, corrected output, model/configuration versions, forecast issue time, and target interval for each installation.
2. On receipt of actual readings, validate them and join them to matching forecast intervals. Track both the measurement timestamp and when the system received the reading.
3. Calculate accuracy metrics and residuals. Flag outages, curtailment, maintenance, clipping, measurement/log anomalies, and missing observations where the available data support those flags. Never train against the system's own predictions as though they were actual generation.
4. Train candidate corrections after an upload or in a nightly backend job when enough new validated observations exist. Forecast refresh remains every user-selected five-minute multiple; learning and deployment have a separate cadence. A nighttime schedule is a proposal for the application, not a Codex automation created by this document.
5. Use rolling chronological training/validation windows, grouped by target day/interval so repeated forecasts of one actual reading cannot leak across splits. Fit all preprocessing on training data only. Use a separation gap where overlapping forecast horizons would leak information. Use a later untouched evaluation period for reporting. [Time-series validation guidance](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
6. Evaluate against the physical baseline and current deployed correction on the same eligible observations and forecast lead times. Promote a candidate only if it passes predefined data-coverage and improvement checks without material degradation in critical operating conditions. Otherwise retain the existing model or physical fallback. Do not tune repeatedly against the final reporting holdout.
7. Version models, log training coverage and comparisons, monitor error/bias changes, and allow rollback. Changes to installation configuration invalidate or revalidate the associated correction model rather than silently reusing it.

Start with a physical baseline for every new installation. Learn site-specific corrections only when validated data support them. Pooling sites requires capacity normalization, site/configuration features, and evaluation on sites held out of training; shared coordinates do not imply identical behavior.

Do not impose a universal number of days as proof that training is ready. Track the number of independent valid days, weather diversity, lead-time coverage, and out-of-sample performance. Display "Collecting data," "Candidate under evaluation," or the active model version and measured validation results. Improvement is an empirical result, not a guaranteed percentage.

All prediction features must be available at forecast issue time: later generation readings, weather observations, and fault logs may be used for diagnosis but not retroactively as forecast inputs. Measured local irradiance and panel temperature are unavailable under the confirmed equipment constraint. For a day-ahead forecast, use forecast/model estimates of future conditions and documented scenarios for unknown operational restrictions.

Distinguish expected unconstrained PV potential from metered operational output. Score the main operational forecast against actuals including real output-limiting events when those are the target, and label a separate healthy-operation score if needed. A physical-conversion correction may exclude faults and curtailment from its training set, but exclusions and counts must be visible; do not hide failures by removing them from the user-facing operational score. Error patterns can suggest possible causes but do not prove a fault or its cause.

## 6. Sunrise-to-sunset forecast contract

Refresh cadence and row duration are independent settings. Cadence is user-configurable in five-minute increments; row duration is always 15 minutes. The grid is tied to the target date's sunrise and sunset at the selected installation, not to the refresh clock. A refresh at 13:05 must not shift the grid to :05/:20/:35/:50.

Round each boundary upward to the next local quarter-hour (:00, :15, :30, :45), leaving an exactly aligned boundary unchanged. Retain actual sunrise/sunset alongside the rounded boundaries. Proposed interpretation of the user's `t+15, t+30, ...` rule:

`t = ceil_to_quarter_hour(actual_sunrise)`

`last_end = ceil_to_quarter_hour(actual_sunset)`

`start_i = t + (i - 1) × 15 minutes`

`end_i = t + i × 15 minutes`, through `last_end`.

Label rows by interval end. For an illustrative sunrise of 06:07 and sunset of 18:02, the rounded table period is 06:15–18:15. Its first row ends at 06:30 and represents 06:15–06:30; its last row ends at 18:15 and represents 18:00–18:15. There are 48 rows in this example, not a fixed row count for all days.

Upward rounding must not move the physical sunrise or sunset. In the last interval, model zero generation after the actual sunset, while reporting mean power over the full 15-minute slot. With the proposed start convention, the short daylight segment from actual sunrise to rounded sunrise is outside the main table; calculate and expose that energy separately so a full-day total is not silently understated. The tomorrow gadget always integrates the full actual daylight period.

Confirmed display behavior:

- Main-table target day: today in the selected installation's time zone. The separate gadget summarizes tomorrow.
- Keep today's entire daylight table visible after every refresh, including completed intervals. Add actual measured generation beside the forecasts for past intervals.
- Retain forecast and actual as separate columns. Preserve issued forecasts; update future predictions without rewriting completed intervals' forecast history. An actual reading arriving later may populate or correct the measurement column with provenance.

For each completed interval, display the most recent saved forecast that was issued at or before that interval's start as the default comparison forecast. Retain all other runs for lead-time analysis. If monitoring starts after an interval has begun and no eligible historical forecast exists, show the forecast comparison as unavailable; do not reconstruct a past prediction with newly available weather. If refresh occurs within an interval, mark it in progress and avoid presenting a recalculation of its elapsed portion as a forecast issued in advance.

Actual generation requires uploaded production logs or an inverter/meter data connection; the weather API cannot measure PV output. Uploads are the initial supported source in this design. Until a matching measurement is available, display "Awaiting data" rather than zero or a modeled estimate. If no physical installation exists yet, the actual column remains unavailable.

Aggregate actual measurements onto exactly the same site, time-zone-aware intervals, units, and AC/DC measurement boundary as the forecast. Derive average actual kW from integrated measured kWh over the interval; for a full 15-minute slot, average kW = kWh / 0.25. Handle cumulative-meter resets and data gaps before aggregation. Flag incomplete measurement coverage rather than extrapolating it as a complete actual value. Deviation columns use actual minus the saved comparison forecast and remain blank if either value is missing. Accuracy percentages follow section 9.

Resample the weather to the quarter-hour grid with documented handling of instantaneous variables, interval means, and accumulated precipitation. Preserve energy when converting interval-mean irradiance; do not simply relabel an API timestamp. Grid resolution does not imply equal native weather resolution.

Each row includes interval start/end, forecast average PV power (kW), forecast interval energy (kWh), actual average PV power (kW) and actual energy (kWh) when available, measurement boundary, deviation (kW and %), accuracy/agreement (%), interval status, relevant weather, and quality flags. For a complete interval: `energy_kWh = average_power_kW × 0.25`. Show uncertainty bounds only if supported by an evaluated method. Clearly label whether a displayed forecast was issued before the interval began. PV DC values and solar AC estimates must not share an unlabeled column or be compared as though measured at the same boundary.

Use timezone-aware UTC instants internally and local dates with offsets in the display. Build intervals using elapsed time and handle offset changes without duplicate or ambiguous identities. Model nighttime generation as zero; missing weather must be marked unavailable rather than replaced by zero. For polar day/night, use an explicit day/night policy rather than inventing sunrise or sunset.

Preserve each forecast run, its configuration version, and its original predictions for later comparison with measurements. The latest view must not erase earlier forecasts.

## 7. Tomorrow gadget contract

Compute a separate forecast for the next calendar date at each installation location. Initialize when configured, then refresh automatically every two hours on a persisted schedule independent of the main daylight forecast. Switching installations displays that installation's own cached summary and schedule. The interface must state the target date, actual sunrise/sunset, rounded display boundaries, last successful update, and next scheduled update. The average-power denominator uses actual daylight duration.

Primary metric: **average potential PV power during tomorrow's daylight hours, in kW**, with the PV DC or modeled solar AC boundary explicitly labeled. Keep unconstrained solar potential and predicted operational harvest separate where battery/load restrictions matter.

`Eday_kWh = integral of predicted PV power at the selected boundary from sunrise to sunset`

`Pday_average_kW = Eday_kWh / daylight_duration_hours`

Also display expected daylight energy (kWh), peak power (kW), and estimated peak time. Integrate partial intervals at sunrise and sunset rather than counting every intersecting interval as a full 15 minutes. Do not average across 24 hours when the label says daylight average.

Illustrative arithmetic only: 36 kWh over 12 daylight hours means a daylight average of 3 kW. This is not a forecast for the user's installation.

At local date rollover, refresh through the independent schedule, including a midnight boundary if wall-clock scheduling is selected. Never relabel an old target date as tomorrow. Handle polar day/night explicitly if those locations are supported.

## 8. Dataset workflow

Confirmed current state: no dataset is available. Start with physical-model forecasts and enable uploads later without requiring a dataset during installation setup.

Accept CSV and Excel XLSX logs with a column-mapping and preview screen. Assign every upload to a saved installation, with explicit confirmation of units, time zone, and whether timestamps denote interval starts, interval ends, or instantaneous samples. Required for generation training: timestamps, measured power or interval energy, and installation identification/specifications. Historical weather may be uploaded or joined from a provider. Retain the source filename, import time, selected column mappings, and validation results.

Automatically archive weather inputs and issued forecasts per installation, and provide CSV export of forecast history. Forecast logs alone are not measured generation and must not be used as evidence that the AI has learned real output. Uploaded production logs or a future meter/inverter connection provide actual training targets. Keep model status visible as "Physics model — no trained AI correction yet" until training and validation have been completed.

Validate sampling intervals, timestamp meaning, duplicates, missing values, units, meter resets, and changes to the installation. Identify outages, curtailment, and inverter downtime so they are not learned as normal weather effects. Never fill missing generation with zero automatically.

For honest forecast evaluation, train and backtest against weather forecasts that were available at each historical issue time. Future observations are not valid substitutes for the weather knowledge available then. Open-Meteo provides an archive of individual forecast runs that can help construct this workflow. [Single Runs API](https://open-meteo.com/en/docs/single-runs-api)

Use chronological train/validation/test periods and report MAE/RMSE in kW, daily energy error, and performance by forecast lead time. Compare the AI model with the physical baseline. Avoid relying on percentage errors near zero production. Calibrate any prediction intervals on held-out data.

## 9. Forecast accuracy and deviation

Add an "Accuracy %" column with the explanation "Forecast agreement with measured output; not a probability." Regression does not have a single universal accuracy percentage. The following bounded agreement score is a proposed, explicitly defined interface metric, accompanied by unbounded error measures so clipping cannot conceal large errors.

For an eligible completed interval, let `F` be its saved forecast and `A` its actual interval-average power in kW:

- `deviation_kW = A - F`: negative means actual generation fell below the forecast.
- `deviation_percent = 100 × (A - F) / A`: denominator is actual generation; state this in the tooltip.
- `absolute_percentage_error = 100 × abs(F - A) / A`.
- `accuracy_percent = max(0, 100 - absolute_percentage_error)`.

Illustrative example: forecast 4.80 kW, actual 5.00 kW → deviation +0.20 kW (+4.0%), absolute percentage error 4.0%, agreement 96.0%. A 4.80 kW forecast for 2.00 kW actual has 140% absolute percentage error and a displayed agreement floor of 0%; retain the 140% error in detail/export.

If actual power is zero or below a configurable low-output threshold, show "N/A — low output" for per-row percentages, while still displaying deviation in kW and energy error. A provisional threshold is the larger of 1% of the corresponding reference capacity and the measurement's documented uncertainty floor: use installed PV STC capacity for a PV DC metric, or inverter AC rating for a solar AC metric. Validate and expose this choice per installation. If the applicable capacity or measurement quality is unknown, identify the assumption instead of silently inventing a measured threshold. Ordinary percentage errors become unstable as the actual denominator approaches zero. [Metric documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html)

Missing, incomplete, or unpaired actuals are "Awaiting data" or "Insufficient data," never 0% or 100% accuracy. A zero/zero interval is not evidence of perfect daylight forecasting and gets no row percentage. Future rows get no retrospective accuracy score.

For today's completed intervals and rolling 7-/30-day summaries, compute duration-weighted absolute error relative to total actual energy over paired valid intervals:

`WAPE_percent = 100 × sum(abs(F_i - A_i) × duration_hours_i) / sum(A_i × duration_hours_i)`

`summary_agreement_percent = max(0, 100 - WAPE_percent)`

Do not average the row percentages and do not substitute `abs(sum(F_i) - sum(A_i))`: the latter can cancel overprediction against underprediction. Unlike per-row percentages, this aggregation includes valid low/zero-output daylight intervals in the numerator. If aggregate actual energy is zero or below the documented aggregate measurement floor, report N/A. Show unbounded WAPE, MAE/RMSE in kW, signed mean bias, and net energy deviation alongside the bounded score.

Label partial-day results "Today so far" and show coverage as valid paired completed intervals / expected completed intervals, together with exclusions and their reasons. The score must not imply full-day coverage when measurements are missing. Surface the selected low-output threshold and the count of row percentages suppressed by it.

Compare like-for-like forecast vintages: the default table uses the last prediction issued at or before interval start; summaries must state this choice. Also report separate performance by lead-time band. Evaluate the tomorrow gadget using a saved forecast issued the previous local day, with an explicit fixed selection rule such as the last successfully published tomorrow summary before local midnight. Revisions made on the target day must not improve the reported day-ahead score retrospectively.

Keep historical accuracy separate from predicted uncertainty. A 90% historical agreement score does not mean there is a 90% probability that the next forecast is correct. Publish future prediction intervals only after an appropriate calibration/evaluation workflow.

## 10. Proposed implementation and dashboard

Proposed components: web dashboard → Python API → weather adapter → pvlib model → optional trained correction model → persistent forecast store. Use an independent background scheduler for both forecast products so jobs can continue while the browser is closed, provided the backend host remains running.

Dashboard areas:

1. Place/installation switcher, Add place, Add installation, editable GPS coordinates, and panel selection.
2. Current forecast status and weather context.
3. Today's full sunrise-to-sunset power chart and quarter-hour log with forecast and actual generation, a variable row count, and completed/in-progress/future status.
4. Tomorrow daylight summary gadget.
5. Scenario simulation and baseline comparison.
6. Dataset upload, accuracy/deviation trends, measurement coverage, and self-learning model status.

Store places, installations and configuration versions, catalog revisions, weather snapshots, forecast runs/rows, daily summaries, dataset imports, observations, and trained-model versions. Scope results and scheduler jobs by installation so switching places cannot mix their data. API failures should preserve the last successful result with an explicit stale status. Use bounded retries, request caching, and locking to prevent overlapping jobs. A fast refresh must not create fabricated new weather information.

## 11. Acceptance checks before calling the application complete

- For sunrise 06:07 and sunset 18:02, rounding produces 06:15 and 18:15; exact quarter-hour boundaries remain unchanged. Under the proposed t+15 interpretation, rows end at 06:30 through 18:15, with 48 rows for this example.
- Refreshing at 13:05 does not shift the quarter-hour grid. Row count changes with daylight duration rather than staying fixed at 24.
- Every positive allowed cadence is a multiple of five minutes and does not change the 15-minute row length.
- The next-day gadget updates on its own two-hour schedule, independently of main-table refreshes.
- All labels, sunrise/sunset calculations, and the meaning of tomorrow use the installation time zone.
- STC power is consistent with the panel count/rating; module efficiency is not applied twice; AC power respects inverter limits.
- Energy equals integrated power, including partial daylight intervals; nighttime and missing-data cases are distinct. Full-day totals include the daylight segment before a rounded-up start even when the main table excludes it.
- Adding a place or installation preserves existing ones. Switching or editing GPS coordinates uses the appropriate time zone and weather while preserving historical configuration associations.
- CSV/XLSX imports are mapped, validated, and associated with the correct installation; missing measurements and forecast-only logs do not enable a falsely labeled trained model.
- A midday refresh retains completed rows and their issued forecast history, updates future predictions, and adds measured actual generation for completed intervals when available. Missing actuals show "Awaiting data," and no retrospectively generated value is presented as a historical forecast.
- Weather resolution, source freshness, forecast history, catalog sources, and model assumptions remain inspectable.
- Forecast and simulation results are clearly distinguished; AI accuracy is supported by held-out measured data before performance claims are made.
- Forecast 4.80 kW versus actual 5.00 kW yields +0.20 kW deviation and 96% row agreement. Near-zero actuals yield no misleading row percentage; large errors remain visible even when agreement is floored at zero.
- Daily summary error is calculated from paired interval errors, not averaged row percentages or net daily energy alone. Missing coverage and operational exclusions are visible.
- Original forecasts are never overwritten to improve reported accuracy. Day-ahead and short-lead performance are evaluated separately with inputs that existed at issue time.
- Candidate model training is distinct from forecast refreshing; only candidates meeting out-of-time validation criteria can replace the active model, and a physical fallback remains available.
- No screen, model, or training workflow requires additional sensors. Irradiance, panel temperature, and soiling estimates are distinguished from measurements; absent actual production data leave accuracy and learned corrections unavailable.
- The Felicity installation uses the confirmed 4 kW AC rating and 24 V battery system. PV DC harvest is not clipped automatically to the AC rating or compared against mixed-source AC load power. Protocol fields, power boundaries, and scaling are verified before actual-generation scoring is enabled.

The main-table display choices are confirmed: today, full daylight period, and actual generation for completed intervals. Initial GPS coordinates and the preferred first implementation deliverable remain to be supplied. Dataset upload is an optional later step and must not block physical-model forecasting. Automatic actual-generation ingestion would additionally require the inverter/meter platform and connection details; until then, actuals can be populated from uploaded production logs.
