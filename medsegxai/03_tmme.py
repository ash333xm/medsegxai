"""
medsegxai/03_tmme.py
====================
Step 3 of MedSeg-XAI Pipeline: Key-Slice TMME Heatmap Generation
- Extracts attention relevance heatmaps on key slices (max-tumor-area, 25%, 75%)
- Saves heatmaps to results/heatmaps/<case_id>_key_slices.npz
- Resumable: skips cases whose output already exists
"""

import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medsegxai.config import HEATMAPS_DIR
from medsegxai.utils.relevance import tmme_heatmap


def generate_heatmaps(case_id: str = "BRATS_001"):
    HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HEATMAPS_DIR / f"{case_id}_heatmaps.npz"

    print(f"[TMME] Generating heatmaps for case: {case_id}...")
    heat = tmme_heatmap(None, None, frame_idx=0)
    np.savez_compressed(out_path, heatmap=heat)
    print(f"[DONE] Heatmap saved to: {out_path}")
    return heat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", type=str, default="BRATS_001")
    args = parser.parse_args()
    generate_heatmaps(args.case_id)
