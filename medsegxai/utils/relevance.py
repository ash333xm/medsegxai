"""
medsegxai/utils/relevance.py
============================
Relevance propagation and attention rollout utilities for TMME explainability.
Contract:
    tmme_heatmap(model, state, frame_idx) -> 2D float array, values in [0, 1]
"""

from typing import Optional, Any
import numpy as np
import torch


def tmme_heatmap(
    model: Any,
    state: Any,
    frame_idx: int,
    target_layer: Optional[str] = None
) -> np.ndarray:
    """
    Computes TMME relevance heatmap on the specified frame.
    Interface Contract:
        tmme_heatmap(model, state, frame_idx) -> 2D float array, values in [0, 1]
    """
    # Placeholder / fallback implementation for Week 2 expansion
    h, w = 512, 512
    # If state contains box prompt, generate localized Gaussian attention
    if isinstance(state, dict) and "prompts" in state and frame_idx in state["prompts"]:
        box = state["prompts"][frame_idx].get("box", [128, 128, 384, 384])
        x0, y0, x1, y1 = [int(v) for v in box]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        sx, sy = max(10.0, (x1 - x0) / 2.5), max(10.0, (y1 - y0) / 2.5)
        yy, xx = np.ogrid[:h, :w]
        heat = np.exp(-0.5 * (((xx - cx) / sx)**2 + ((yy - cy) / sy)**2))
        return heat.astype(np.float32)

    # Default normalized spatial grid
    yy, xx = np.ogrid[:h, :w]
    heat = np.exp(-0.5 * (((xx - 256) / 80.0)**2 + ((yy - 256) / 80.0)**2))
    return heat.astype(np.float32)
