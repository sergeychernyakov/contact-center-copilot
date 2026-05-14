"""Generate a realistic sample Excel for testing.

Run: python scripts/generate_sample_data.py
Output: data/sample_diagnostic.xlsx
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)
N = 90  # daily rows for one quarter


def make_performance() -> pd.DataFrame:
    dates = pd.date_range("2025-07-01", periods=N, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "FCR": RNG.normal(0.68, 0.04, N).clip(0.5, 0.85),  # 68% mean
            "AHT_seconds": RNG.normal(420, 35, N).clip(300, 600),  # 7 min mean
            "CSAT": RNG.normal(0.81, 0.03, N).clip(0.6, 0.95),
            "AbandonRate": RNG.normal(0.07, 0.015, N).clip(0.02, 0.15),
            "ServiceLevel_80_20": RNG.normal(0.78, 0.04, N).clip(0.5, 0.95),
            "CallVolume": RNG.integers(3500, 4800, N),
        }
    )


def make_agent_stats() -> pd.DataFrame:
    agents = [f"AGT-{i:04d}" for i in range(1, 121)]
    return pd.DataFrame(
        {
            "AgentID": agents,
            "AgentUtilization": RNG.normal(0.72, 0.08, len(agents)).clip(0.4, 0.95),
            "Occupancy": RNG.normal(0.85, 0.05, len(agents)).clip(0.6, 0.98),
            "CallsHandled": RNG.integers(800, 1500, len(agents)),
            "AvgAHT_seconds": RNG.normal(420, 60, len(agents)).clip(280, 700),
        }
    )


def make_hr_costs() -> pd.DataFrame:
    """Deliberately irrelevant sheet — the Schema Inspector should skip this."""
    months = pd.date_range("2025-01-01", periods=12, freq="MS")
    return pd.DataFrame(
        {
            "Month": months,
            "Payroll_USD": RNG.normal(580000, 30000, 12),
            "Benefits_USD": RNG.normal(120000, 8000, 12),
            "Office_USD": RNG.normal(45000, 5000, 12),
        }
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_path = root / "data" / "sample_diagnostic.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        make_performance().to_excel(writer, sheet_name="Q3_Performance", index=False)
        make_agent_stats().to_excel(writer, sheet_name="Agent_Stats", index=False)
        make_hr_costs().to_excel(writer, sheet_name="HR_Costs", index=False)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
