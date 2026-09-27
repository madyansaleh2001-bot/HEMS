import csv
from pathlib import Path

from pvforecast.cli import main

EXAMPLE = str(Path(__file__).resolve().parents[1] / "examples" / "example_installation.json")


def test_today_simulation_is_labelled_and_awaits_actuals(capsys, tmp_path):
    out_csv = tmp_path / "today.csv"
    assert main(["today", "--config", EXAMPLE, "--clear-sky", "--now", "2026-09-27T13:05", "--csv", str(out_csv)]) == 0
    text = capsys.readouterr().out
    assert "SIMULATION" in text and "not a weather forecast" in text
    assert "Awaiting actual data" in text
    assert "Physics model — no trained AI correction yet" in text
    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert len(rows) == 48
    assert rows[0]["label"] == "07:15" and rows[0]["interval_start"].startswith("2026-09-27T07:00")


def test_tomorrow_simulation(capsys):
    assert main(["tomorrow", "--config", EXAMPLE, "--clear-sky", "--now", "2026-09-27T13:05"]) == 0
    text = capsys.readouterr().out
    assert "Tomorrow: 2026-09-28" in text
    assert "Daylight-average potential power" in text and "every 2 h" in text
