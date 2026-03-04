"""Magnitude-based pruning masks (unstructured and N:M structured)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


@torch.no_grad()
def magnitude_mask(W: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Unstructured magnitude pruning mask.

    Args:
        W: weight matrix (out, in)
        sparsity: fraction of weights to prune (0..1). e.g., 0.5 means 50% zeros.

    Returns:
        P: bool mask, True means keep (non-zero).
    """
    if W.ndim != 2:
        raise ValueError(f"magnitude_mask expects 2D weight, got {W.shape}")
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0,1), got {sparsity}")
    numel = W.numel()
    k_prune = int(math.floor(numel * sparsity))
    if k_prune <= 0:
        return torch.ones_like(W, dtype=torch.bool)
    if k_prune >= numel:
        return torch.zeros_like(W, dtype=torch.bool)

    scores = W.abs().flatten()
    # kthvalue uses 1-based k.
    kth = torch.kthvalue(scores, k_prune).values
    # keep strictly greater than kth (matches paper P_ij = I(M_ij > h))
    P = (W.abs() > kth)
    # If ties cause slightly fewer pruned, that's acceptable per the paper's percentile definition.
    return P


@torch.no_grad()
def nm_structured_mask(
    W: torch.Tensor,
    n: int,
    m: int,
    *,
    dim: int = 1,
    score: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """N:M structured mask.

    Keeps top-n weights (by |W| or custom score) within each group of size m along dimension `dim`
    (typically dim=1 i.e. along input features per output row).

    Args:
        W: (out, in)
        n: number of kept elements per group
        m: group size
        dim: grouping dimension (0 for output dim, 1 for input dim)
        score: optional score tensor same shape as W; if None uses |W|.

    Returns:
        P: bool mask, True means keep.
    """
    if W.ndim != 2:
        raise ValueError(f"nm_structured_mask expects 2D weight, got {W.shape}")
    if not (0 < n <= m):
        raise ValueError(f"Require 0 < n <= m, got n={n}, m={m}")
    if dim not in (0, 1):
        raise ValueError(f"dim must be 0 or 1, got {dim}")

    if score is None:
        score = W.abs()
    else:
        if score.shape != W.shape:
            raise ValueError("score must have same shape as W")

    # Move grouping dimension to last for easy reshape
    if dim == 0:
        score_t = score.transpose(0, 1)  # (in, out)
    else:
        score_t = score  # (out, in)

    a, b = score_t.shape  # a=out (or in), b=in (or out) after transpose logic
    if b % m != 0:
        raise ValueError(f"Dimension size {b} is not divisible by m={m}. For LLaMA layers this should be divisible.")

    grouped = score_t.reshape(a, b // m, m)  # (..., groups, m)
    # topk over last dim
    topk = torch.topk(grouped, k=n, dim=-1, largest=True, sorted=False).indices  # (..., groups, n)
    mask_grouped = torch.zeros_like(grouped, dtype=torch.bool)
    # scatter True at selected indices
    mask_grouped.scatter_(-1, topk, True)
    mask_t = mask_grouped.reshape(a, b)

    if dim == 0:
        P = mask_t.transpose(0, 1)
    else:
        P = mask_t
    return P
