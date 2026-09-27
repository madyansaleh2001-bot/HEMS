"""House 12 — Trina 505 W. Open in VS Code and press Run (▶ "Run Python File").

Writes pv_house_outputs/model_datasets/house_12_model_dataset.csv: SIMULATED available PV
potential, not measured generation. Needs simulate_pv.py and simulation_config.json in this
same folder and the packages from requirements.txt.
"""
from pathlib import Path

from simulate_pv import run_single_house

INPUT_DIR = Path(r"C:\Users\MI Electronics\Downloads")  # folder with "Weather House 12.csv" (only read)
OUTPUT_DIR = INPUT_DIR / "pv_house_outputs"              # results go here, never over the originals

# House 12 note: gamma -0.0034 /degC: Trina DE18M(II) family sheet (installed revision unconfirmed)
HOUSE = {
    "house_id": 12,
    "panel_identification": "Trina 505 W",
    "panel_count": 10,                              # illustrative scenario
    "panel_stc_power_w": 505,                       # Pmax per panel, W at STC (your table)
    "tilt_deg": 31,                                 # your table
    "azimuth_deg": 180,                             # assumption: south
    "gamma_pdc_per_c": -0.0034,                     # Pmax temperature coefficient, fraction per degC
    "inverter_type": "string",                      # internal only, not exported
    "inverter_unit_count": 1,                       # 1 = string/hybrid; microinverters: one per panel
    "inverter_ac_power_w_per_unit": 4000,           # W per unit (illustrative scenario)
    "inverter_nominal_efficiency": 0.97,            # fraction (generic assumption)
}

if __name__ == "__main__":
    raise SystemExit(run_single_house(HOUSE, INPUT_DIR, OUTPUT_DIR))
