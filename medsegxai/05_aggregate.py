"""
medsegxai/05_aggregate.py
=========================
Step 5 of MedSeg-XAI Pipeline: Aggregate Cohort Reporting & Figures
- Aggregates Dice scores, timing, and MPRT audit verdicts into summary tables
- Generates cohort visualizations in results/figures/
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medsegxai.config import TABLES_DIR, FIGURES_DIR, DICE_RESULTS_CSV


def aggregate_results():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if DICE_RESULTS_CSV.exists():
        df = pd.read_csv(DICE_RESULTS_CSV)
        print("Cohort Dice Summary:")
        print(df.to_string())
    else:
        print("No dice results found to aggregate yet.")


if __name__ == "__main__":
    aggregate_results()
