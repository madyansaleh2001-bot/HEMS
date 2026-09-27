# Historical PV simulation — Weather House 1–13

This package turns the 13 source weather CSVs (Spain, 15-minute, 2021-01-01 to 2024-04-30)
into one 20-column modelling dataset per house using pvlib.

> **All power values are simulated available PV potential, not measured generation.** They
> include no battery, load, grid or MPPT curtailment. Because there are no measured targets or
> saved forecasts, they cannot show real forecast accuracy.

## Run it in VS Code (one file per house)

1. Unzip the package into a folder, e.g. `C:\Users\MI Electronics\Downloads\pv_simulation`, and open that
   folder in VS Code (**File → Open Folder**).
2. **Terminal → New Terminal**, then (Python 3.12 must be installed):
   ```bat
   py -3.12 -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```
3. **Ctrl+Shift+P → "Python: Select Interpreter"** → pick `.venv`.
4. Open `house_01.py` … `house_13.py` and press **▶ Run Python File**. Each one audits its
   `Weather House N.csv`, simulates it and validates the result. It writes to
   `Downloads\pv_house_outputs\`:
   - `model_datasets\house_NN_model_dataset.csv`
   - `metadata\house_NN_audit.json`
   - `metadata\house_NN_validation.json`
   - `metadata\house_NN_run_metadata.json`
   - `house_NN_energy_summary.csv`

   Rerunning a house replaces only that house's own files.
5. `run_all_houses.py` runs all 13 into `Downloads\pv_simulation_output\` and builds `pv_model_datasets.zip`.

`INPUT_DIR` and `OUTPUT_DIR` are at the top of each file. The house settings are in its `HOUSE`
block. If you edit them, the change is printed and recorded in the metadata as an edited
scenario. Every house file uses `simulate_pv.py` and `simulation_config.json` from the same folder.

## Run it from a terminal

```bat
:: Windows, Python 3.12
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python simulate_pv.py --input-dir "C:\Users\MI Electronics\Downloads" --output-dir pv_simulation_output --energy-summary --zip
```

```bash
# Linux/macOS
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python simulate_pv.py --input-dir /path/to/weather --output-dir pv_simulation_output --energy-summary --zip
```

The full set of 13 files × 116,641 rows takes about 1.5 minutes. The exit code is 0 only when every
requested house passes both the audit and the validation.

| Option | Purpose |
|---|---|
| `--input-dir` | Folder with `Weather House N.csv` (opened read-only; must differ from `--output-dir`) |
| `--output-dir` | New output folder (default `pv_simulation_output`) |
| `--config` | Editable settings (default `simulation_config.json` next to the script) |
| `--houses 1,4,8` | Process a subset |
| `--audit-only` | Audit the sources without simulating |
| `--energy-summary` | Also write `energy_summary.csv` (kept separate from the datasets) |
| `--reference-dir DIR` | Compare DC/AC with earlier `house_XX_model_dataset.csv` files (tolerance 1e-7 W) |
| `--zip` | Bundle everything into `pv_model_datasets.zip` |
| `--overwrite` | Reuse an output folder: first removes that folder's earlier `model_datasets/`, `metadata/`, `energy_summary.csv` and ZIP, so stale results can never be mistaken for current ones |

## Outputs

```
pv_simulation_output/
  model_datasets/house_01_model_dataset.csv … house_13_model_dataset.csv   (identical 20-column header)
  metadata/run_metadata.json        provenance statement, versions, settings and their sources, SHA-256 of sources and outputs
  metadata/audit_report.json        per-file audit: defects, warnings, diagnostics
  metadata/validation_report.json   per-house validation results
  energy_summary.csv                optional, separate from the datasets
  README.md  schema.json  simulate_pv.py  simulation_config.json  requirements.txt
  pv_model_datasets.zip
```

Final header (exact): see `schema.json`. `Fill_Flag`, `Record_Type`, `POA_Global_Wm2`,
`PV_Cell_Temperature_C` and `Inverter_Type` are not exported. POA irradiance and cell temperature
are still computed internally, and the inverter type stays in the configuration and metadata.

## How weather becomes DC and then AC power

The following steps run at every source timestamp independently. There is no resampling,
averaging or interpolation.

1. **Solar geometry.** The source `Sun Zenith (rad)` and `Sun Azimuth (rad)` are converted to degrees for pvlib only. No sun angles are recomputed from coordinates, and none are known.
2. **Extraterrestrial DNI.** `pvlib.irradiance.get_extra_radiation(timestamps)` with pvlib's default settings. The day of year comes from the preserved timestamp label.
3. **Front-side POA.** `get_total_irradiance(model="haydavies")` uses the house tilt, a 180° azimuth, the source GHI/DNI/DHI, the source `Surface Albedo` for ground reflection, and the source angles. POA is set to at least 0, and to 0 whenever zenith ≥ 90°. Bifacial rear gain is 0.
4. **Cell temperature.** `sapm_cell(POA, Temperature, Wind Speed)` with the `open_rack_glass_polymer` parameters. This is a generic approximation applied to every house.
5. **DC.** `pvwatts_dc` gives `P_STC · POA/1000 · (1 + γ·(Tcell − 25))`, clipped at 0, then × 0.90 for the 10 % aggregate DC loss. This is `PV_Power_Generation_W`. DC is not capped at the array STC rating.
6. **AC.** Each of the `n` inverter units receives `DC/n`. `pvlib.inverter.pvwatts(DC/n, pdc0 = Pac_unit/η, eta_inv_nom = η, eta_inv_ref = 0.9637)` applies the part-load efficiency curve and clips at `Pac_unit`. The result × `n`, floored at 0, is `PV_AC_Power_W`. `pdc0` here is the inverter's DC reference input, not the array STC rating.

Cloud type, pressure, humidity, dew point, ozone and precipitable water are not applied as extra
losses; the source irradiance already reflects the weather.

## Source-supplied settings vs scenario assumptions

| Setting | Status |
|---|---|
| Panel models, Pmax per panel, tilts | Your supplied tables. Houses 4 and 8 use **445 W**, which you selected, although the model name says 440 W. |
| γ (Pmax temperature coefficient) | Trina (Houses 1, 9, 12): −0.0034 /°C from the DE18M(II) family sheet; the installed revision is unconfirmed. All others: **assumed** −0.0035 /°C. |
| Azimuth 180° (south) for every house | **Scenario assumption** |
| Panel counts, inverter types, unit counts, AC ratings | **Authorized illustrative scenarios.** They are not the real Spanish installations. |
| Inverter nominal efficiencies (0.93 / 0.97 / 0.965) | **Generic assumptions**, not manufacturer curves |
| 10 % aggregate DC loss; SAPM open-rack thermal model | **Scenario assumptions** |
| Units (W/m², °C, m/s, %, albedo fraction, radians) | **Assumed** and recorded in the metadata |
| Coordinates, time zone, timestamp meaning | **Unconfirmed**; not used, and timestamps are not shifted |

Module-specific notes are kept in `simulation_config.json` and copied into `run_metadata.json`:
- The JA Solar bifacial panels are simulated front side only.
- The Xunzel module's listed efficiency conflicts with its rating; the model uses 425 W with no correction.
- Microinverters get equal per-panel power, with no shading or MPPT benefit.

## Audit: nothing is filled, dropped or replaced

Every source column is checked for:
- empty or whitespace-only cells, and blank lines (a blank line means a missing record);
- missing-value tokens (`NA`, `null`, `#N/A` …);
- non-numeric values, infinities, and placeholder candidates (−9999, −999, −99 …; 999 is allowed because it is a valid pressure);
- a wrong number of fields in a row, and duplicate or missing column names;
- invalid, duplicate or non-monotonic timestamps, gaps (missing whole rows) and off-grid timestamps.

Night-time zeros and 0 °C are valid. Any defect blocks only that house's output, and the report lists
the file, column, line and timestamp. The row count and first and last timestamps are compared with
the earlier audit (116,641 rows, 2021-01-01 00:00:00 to 2024-04-30 00:00:00); a mismatch is
reported. The source `Fill Flag` is audited but never used or exported. A complete file does not
prove the provider never filled earlier gaps.

The audit also reports diagnostics that never change the data:
- value-range counts;
- records with zenith ≥ 90° but GHI > 0 (irradiance the horizon rule drops);
- the median clock time of the daily minimum zenith (solar noon) in winter and summer. About 60 minutes of difference suggests local time with daylight saving; about 0 suggests UTC or standard time;
- the median solar azimuth at solar noon. It should be about 180° if the source uses pvlib's north = 0, clockwise convention. Otherwise a warning is raised.

## Validation performed on every output file

- Exact header and column order. The five removed columns and obvious aliases are absent, and there is no index column.
- The row count and every timestamp match the source, including the final endpoint. Timestamps are `YYYY-MM-DD HH:MM:SS` with no offset.
- The source weather and solar-angle values are preserved **bit-for-bit**. Sources are parsed with exactly rounded decimal conversion, and outputs are re-read with `float_precision="round_trip"`. pandas' default fast parser can change the last bit.
- `Array_STC_Power_W = Panel_Count × Pmax`; `Inverter_AC_Rated_Power_W = units × per-unit rating`.
- DC and AC are finite and non-negative, and both are 0 when zenith ≥ 90°. AC is never above the aggregate rating or above DC.
- A hand-written PVWatts DC and inverter equation matches pvlib to within 1e-7 W.
- Recomputing DC and AC from the exported file alone (plus the source albedo) reproduces them. This shows that dropping POA, cell temperature and inverter type changed only the schema.
- Optionally, DC and AC are compared with reference outputs (`--reference-dir`).

Floating-point note: at clipping, pvlib computes the AC limit as `η × (Pac/η)`. That can round to
the rating ± about 5e-13 W; for example, 0.97 × (4000 / 0.97) = 4000.0000000000005. The validator
allows 1e-9 W and records the exact largest excess. Values are not clamped, so the physics stays
exactly as specified.

## Energy summary (optional)

`energy_summary.csv` treats source times as instantaneous samples, because their averaging
convention is unconfirmed. It uses trapezoids: `((p[:-1] + p[1:]) / 2) × 0.25 / 1000` kWh. N rows
give N − 1 intervals (116,640 per file), and no energy is added after the final endpoint. Each
interval belongs to the year of its start. 2021–2023 are labelled complete calendar years and
January–April 2024 is labelled partial.

## Key functions (`simulate_pv.py`)

| Function | Role |
|---|---|
| `load_config` | Loads and checks `simulation_config.json` (13 houses, physics settings) |
| `audit_file` | Structural, cell and timestamp audit. Returns exactly parsed values, or `None` if there are defects |
| `diagnostics` | Informational range, solar-noon and azimuth-convention checks |
| `simulate` | Steps 1–6 above; returns DC/AC plus internal POA and cell temperature |
| `build_output` | Selects exactly `FINAL_COLUMNS` |
| `validate_output` | All checks listed above, run on the written file |
| `energy_summary` | Optional trapezoidal yearly energy |

Tests: `pip install pytest && pytest tests` (they use **synthetic** weather files from
`tests/synthetic_weather.py`, never the real data).
