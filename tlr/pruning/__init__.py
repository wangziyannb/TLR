"""Pruning mask generators.

We implement:
- Magnitude pruning (unstructured) and N:M structured magnitude.
- Wanda-style score pruning (requires calibration activations).

For SparseGPT masks: this repo supports loading external masks, see scripts/apply_prune_refine.py.
"""

from .magnitude import magnitude_mask, nm_structured_mask
from .wanda import WandaStats, collect_wanda_stats, wanda_mask

__all__ = [
    "magnitude_mask",
    "nm_structured_mask",
    "WandaStats",
    "collect_wanda_stats",
    "wanda_mask",
]
