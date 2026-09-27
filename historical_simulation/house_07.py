"""House 7 — Tensite Full Black 405 W. Open in VS Code and press Run (▶ "Run Python File").

Writes pv_house_outputs/model_datasets/house_07_model_dataset.csv: SIMULATED available PV
potential, not measured generation. Needs simulate_pv.py and simulation_config.json in this
same folder and the packages from requirements.txt.
"""
from pathlib import Path

from simulate_pv import run_single_house

INPUT_DIR = Path(r"C:\Users\MI Electronics\Downloads")  # folder with "Weather House 7.csv" (only read)
OUTPUT_DIR = INPUT_DIR / "pv_house_outputs"              # results go here, never over the originals

HOUSE = {
    "house_id": 7,
    "panel_identification": "Tensite Full Black 405 W",
    "panel_count": 11,                              # illustrative scenario
    "panel_stc_power_w": 405,                       # Pmax per panel, W at STC (your table)
    "tilt_deg": 39,                                 # your table
    "azimuth_deg": 180,                             # assumption: south
    "gamma_pdc_per_c": -0.0035,                     # Pmax temperature coefficient, fraction per degC
    "inverter_type": "string",                      # internal only, not exported
    "inverter_unit_count": 1,                       # 1 = string/hybrid; microinverters: one per panel
    "inverter_ac_power_w_per_unit": 3600,           # W per unit (illustrative scenario)
    "inverter_nominal_efficiency": 0.97,            # fraction (generic assumption)
}

if __name__ == "__main__":
    raise SystemExit(run_single_house(HOUSE, INPUT_DIR, OUTPUT_DIR))
