# HEMS — AI Forecasting: PV power generation forecasting and simulation

Forecasts the output of fixed PV installations at named GPS places. It covers today's
sunrise-to-sunset table in 15-minute rows and a separate summary for tomorrow. The design
supports adding actual generation and a validated self-learning correction later.

Requirements: [`docs/engineering-specification.md`](docs/engineering-specification.md) and
[`docs/felicity-dataset-specification.md`](docs/felicity-dataset-specification.md).
Decisions made during implementation: [`docs/implementation-notes.md`](docs/implementation-notes.md).

## Current status (honest summary)

| Area | State |
|---|---|
| Forecast core (`backend/`, Python + pvlib) | **Implemented and unit-tested** — quarter-hour daylight grid, physical baseline, today table, tomorrow gadget, accuracy/deviation metrics, table composition |
| Weather | Open-Meteo adapter implemented and tested against a fixture only. A live request **has not been tested** (the development container's network policy blocks the API). Clear-sky *simulation* works offline |
| Actual generation | **None.** No dataset, no uploads, no inverter connection. Every completed row shows "Awaiting actual data" and accuracy is unavailable |
| AI correction | **None trained.** Model status reads "Physics model — no trained AI correction yet" |
| Felicity IVEM4024-II | Preset only: 4 kW AC and 24 V battery (user-confirmed). RS232 protocol, firmware and PV-power register mapping are unverified; no collector exists |
| Persistence, scheduler, API, dashboard, CSV/XLSX import | Not started |
| Site configuration | **Not supplied yet.** `backend/examples/example_installation.json` is a labelled placeholder, *not* the user's site or panels |

## Quick start

```bash
cd backend
pip install -e ".[dev]"
python -m pytest

# Quarter-hour grid for a date
pvforecast grid --config examples/example_installation.json --date 2026-09-27

# Today's table — SIMULATED cloud-free weather (not a forecast); issue time before sunrise shows every row
pvforecast today --config examples/example_installation.json --clear-sky --now 2026-09-27T05:00 --csv today.csv

# Tomorrow gadget
pvforecast tomorrow --config examples/example_installation.json --clear-sky

# Live weather (requires network access to api.open-meteo.com; untested so far)
pvforecast today --config my_site.json --open-meteo --save-weather wx.json
pvforecast today --config my_site.json --weather-json wx.json      # replay an archived response
```

## Conventions implemented

- **Grid:** `t = ceil15(actual sunrise)`, `last_end = ceil15(actual sunset)`. Each row is `[t+(i−1)·15, t+i·15)` and is labelled by its end time. Sunrise 06:07 and sunset 18:02 give rows ending 06:30 through 18:15, which is 48 rows. The grid depends only on the date and place, never on the refresh time. The daylight between actual sunrise and the rounded start is reported separately and included in the full-day total.
- **Boundaries:** the default forecast is *unconstrained PV potential at the inverter's PV DC input*. The modelled solar AC estimate is a separate column, clipped at the 4 kW AC rating. DC harvest is never clipped to the AC rating.
- **Time:** tz-aware UTC internally, with elapsed-time interval arithmetic. Daylight-saving days have no duplicate or missing rows. Polar day and polar night have explicit policies.
- **Forecast history:** a run never gives values to rows that had already finished when it was issued. A completed row is compared with the latest run issued at or before the row started.
- **Accuracy:** implements spec §9 exactly. Row agreement is `max(0, 100 − APE)` with actual power as the denominator. Rows with low output show "N/A". Summaries use duration-weighted WAPE, never averaged row percentages. Agreement is not a probability.

## Repository layout

```
backend/
  src/pvforecast/   config · timegrid · sun · weather · openmeteo · physics · products · accuracy · view · cli
  tests/            acceptance checks from spec §11 plus unit tests
  examples/         placeholder configuration (not the user's site)
docs/               specifications and implementation notes
```
