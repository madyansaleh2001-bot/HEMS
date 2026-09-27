# Implementation notes — forecast core (deliverable 1)

This file records decisions made while implementing the specification, and which of them
are engineering choices still open to review. It is not a list of confirmed user requirements.

## Scope of deliverable 1

A Python library and CLI in `backend/` that produce, for one installation:

1. today's sunrise-to-sunset table on the quarter-hour grid (spec §6);
2. the tomorrow daylight summary (spec §7);
3. per-row and summary accuracy metrics (spec §9), and the composition of saved forecasts
   with actuals into the displayed table.

Persistence, scheduling, the HTTP API, the dashboard, dataset import, the inverter collector
and model training are not part of this deliverable.

## Engineering choices (open to review)

| Topic | Choice | Reason |
|---|---|---|
| Cell temperature | pvlib **SAPM** model with pvlib's per-mounting parameter sets (`open_rack_*`, `close_mount_glass_glass`, `insulated_back_glass_polymer`) | SAPM is defined for wind speed at 10 m, the height the weather API reports. pvlib ships documented presets per mounting style. The spec cites Faiman as a reference, but pvlib has no per-mounting Faiman presets. |
| Transposition | Perez (pvlib `allsitescomposite1990`), physical IAM on beam only | Standard pvlib practice. The diffuse IAM is not applied (small effect). |
| DC model | PVWatts DC plus PVWatts multiplicative losses | Matches spec §5. Loss defaults are pvlib's PVWatts defaults, except **availability = 0 %**: outages are flagged in scoring rather than averaged into every forecast. Every default is shown as an assumption. |
| Weather at 1-minute steps | GHI interval means spread in proportion to clear-sky GHI inside each native interval, which is **exactly energy-preserving**. The provider diffuse fraction is held constant per interval and DNI comes from closure. Beam is not resolved above 87° zenith. Temperature and wind are interpolated linearly. | Satisfies "preserve energy when converting interval-mean irradiance" without double-counting cloud effects. Clear-sky is used only as the within-interval shape. |
| Daylight mask | Power is zero outside actual sunrise to sunset (pvlib SPA, apparent horizon) | The spec asks for zero generation after actual sunset. It also keeps the table, the pre-table segment and the tomorrow integral consistent. |
| Integration | 1-minute midpoint bins. Partial minutes at sunrise and sunset are weighted by overlap. | Integrates partial daylight intervals rather than counting them as full slots. |
| Interval status at exactly its start | `FUTURE` (nothing has elapsed) | This makes a run issued exactly at an interval start eligible as that interval's comparison forecast ("issued at or before interval start"). |
| Completed rows in a new run | No forecast value; flagged `completed_before_issue` | Never reconstruct a past prediction from newer weather. |
| In-progress rows | Value kept and flagged `in_progress_at_issue`, but not eligible as the comparison forecast | Spec §6: the elapsed portion must not be shown as a forecast issued in advance. |
| Tomorrow peak | Highest 15-minute mean, reported with its interval | Consistent with the table. A 1-minute peak from hourly weather would overstate precision. |
| Low-output threshold | 1 % of STC capacity (DC) or of the 4 kW AC rating (AC), raised to the measurement floor when one is known. It is labelled provisional. | Spec §9. The measurement floor for the Felicity telemetry is unknown. |
| Missing temperature coefficient | Placeholder −0.35 %/°C, always reported as an assumption | Only used for manual entries without a datasheet value. |
| Felicity PV input limit | Not applied. The 6,000 W product-page value is recorded in a note only. | The spec says to validate revision-specific limits first. |
| DST fall-back | Rows are identified by UTC instant. Repeated wall-clock labels get a `(UTC±hhmm)` suffix. | No duplicate or ambiguous row identities. |
| Polar day/night | Polar day covers the whole local day. Polar night has no rows, zero energy and no average. Days with a sunrise but no sunset (or the reverse) raise an explicit error. | An explicit policy instead of invented sunrise or sunset times. |

## Things verified only by tests or simulation

- The physics numbers have been checked for plausibility against clear-sky simulations only.
  No comparison with measured generation has been possible because no data exist yet.
- The Open-Meteo parser was tested against a synthetic fixture that follows the documented
  layout (`hourly_units` strings such as `W/m²`, `m/s`). The first live request should be checked
  manually; the parser rejects unexpected units rather than guessing.

## Proposed next deliverables

1. **Persistence and scheduler.** SQLite (SQLAlchemy) tables for places, installations and
   configuration versions, weather snapshots (raw responses archived), immutable forecast runs and
   rows, and tomorrow summaries. Add an APScheduler job per installation for the table (any 5-minute
   multiple), a separate 2-hour job for tomorrow, locking and a stale-result status.
2. **HTTP API and dashboard.** FastAPI endpoints plus a web dashboard: place and installation
   switcher, today table and chart, tomorrow gadget, CSV export.
3. **Production-log import.** CSV/XLSX mapping with a preview; time-weighted aggregation to the
   grid with coverage; `ActualInterval` records feeding `view.compose_today_view`.
4. **Felicity collector.** Read-only and RS232, blocked on the protocol document or a verified
   export for the unit's firmware.
5. **Residual learning.** Blocked on validated measured data (spec §5.2).
