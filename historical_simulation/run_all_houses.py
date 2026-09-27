"""All 13 houses in one go. Open in VS Code and press Run (▶ "Run Python File").

Writes pv_simulation_output/ with the 13 datasets, combined metadata, energy_summary.csv and
pv_model_datasets.zip. Earlier results in that output folder are replaced (only files this
script generates). Simulated available PV potential, not measured generation.
"""
from pathlib import Path

from simulate_pv import main

INPUT_DIR = Path(r"C:\Users\MI Electronics\Downloads")  # folder with "Weather House 1.csv" ... "Weather House 13.csv"
OUTPUT_DIR = INPUT_DIR / "pv_simulation_output"

if __name__ == "__main__":
    raise SystemExit(main([
        "--input-dir", str(INPUT_DIR),
        "--output-dir", str(OUTPUT_DIR),
        "--energy-summary",
        "--zip",
        "--overwrite",
    ]))
