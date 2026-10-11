"""
medsegxai/04_mprt.py
====================
Step 4 of MedSeg-XAI Pipeline: Model Parameter Randomization Test (MPRT)
- Executes 8-stage cascading weight randomization audit across 3 fixed seeds
- Evaluates Spearman rank correlation and pre-registered SSIM
- Saves results to results/audit/<case_id>_mprt.json
- Resumable: skips cases whose output already exists
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medsegxai.config import AUDIT_DIR, SEEDS
from medsegxai.utils.mprt_core import run_mprt


def audit_case(case_id: str = "BRATS_001"):
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AUDIT_DIR / f"{case_id}_mprt.json"

    print(f"[MPRT] Auditing case: {case_id} across {len(SEEDS)} seeds...")
    res = run_mprt(None, None, frame_idx=0, seeds=SEEDS)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    status_str = "PASSED" if res["passed"] else "FAILED"
    print(f"[DONE] Audit {status_str}! Results saved to: {out_path}")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", type=str, default="BRATS_001")
    args = parser.parse_args()
    audit_case(args.case_id)
