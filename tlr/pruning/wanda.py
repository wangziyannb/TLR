"""Wanda-style pruning (weight * activation statistics).

The paper uses Wanda pruning (Sun et al., 2023) with 128 calibration sequences from C4
and then applies the proposed refinement as a post-processing step.

This file implements the activation-statistics part to reproduce Wanda masks:
    score_ij = |W_ij| * s_j
where s_j is computed from calibration activations of the layer input:
    s_j = sqrt(mean(x_j^2))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from .magnitude import nm_structured_mask


@dataclass
class WandaStats:
    """Accumulated Wanda statistics for a set of Linear modules."""

    # For each module name: sum of squares per input feature (shape: [in_features])
    sum_sq: Dict[str, torch.Tensor]
    # For each module name: number of activation rows accumulated
    count: Dict[str, int]

    def scaler(self, name: str) -> torch.Tensor:
        if name not in self.sum_sq:
            raise KeyError(name)
        c = self.count.get(name, 0)
        if c <= 0:
            raise ValueError(f"No stats collected for {name}")
        return torch.sqrt(self.sum_sq[name] / float(c))


@torch.no_grad()
def collect_wanda_stats(
    model: nn.Module,
    name_to_linear: Dict[str, nn.Linear],
    calibration_batches: Iterable[Dict[str, torch.Tensor]],
    *,
    device: torch.device,
    amp_dtype: torch.dtype = torch.float16,
    use_cache: bool = False,
    max_batches: Optional[int] = None,
) -> WandaStats:
    """Collect activation statistics needed for Wanda pruning.

    Args:
        model: HF causal LM model.
        name_to_linear: dict of {qualified_name: nn.Linear}.
        calibration_batches: iterable of tokenized batches (must contain input_ids and attention_mask).
        device: where to run the forward pass.
        amp_dtype: autocast dtype for faster inference.
        use_cache: whether to enable KV cache (should be False for full activation collection).
        max_batches: optionally stop after N batches.

    Returns:
        WandaStats with per-layer sum_sq and counts.
    """
    model.eval()
    sum_sq = {name: torch.zeros((lin.in_features,), dtype=torch.float64, device="cpu") for name, lin in name_to_linear.items()}
    count = {name: 0 for name in name_to_linear.keys()}

    handles = []

    def make_hook(name: str):
        def hook(module: nn.Module, inputs, output):
            # inputs[0] is activation: (..., in_features)
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            x = x.detach()
            # Flatten tokens
            x = x.reshape(-1, x.shape[-1]).to(dtype=torch.float32)
            # Accumulate on CPU to avoid GPU memory growth
            sum_sq[name] += (x * x).sum(dim=0).to(dtype=torch.float64, device="cpu")
            count[name] += x.shape[0]
        return hook

    for name, lin in name_to_linear.items():
        handles.append(lin.register_forward_hook(make_hook(name)))

    # Run calibration forward passes
    with torch.no_grad():
        for bi, batch in enumerate(calibration_batches):
            if max_batches is not None and bi >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=str(device).split(":")[0], dtype=amp_dtype, enabled=(device.type == "cuda")):
                model(**batch, use_cache=use_cache)

    for h in handles:
        h.remove()

    # Convert sum_sq to float32 for storage
    sum_sq_f32 = {k: v.to(dtype=torch.float32) for k, v in sum_sq.items()}
    return WandaStats(sum_sq=sum_sq_f32, count=count)


@torch.no_grad()
def wanda_mask(
    W: torch.Tensor,
    scaler: torch.Tensor,
    *,
    sparsity: Optional[float] = None,
    nm: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """Build Wanda pruning mask for a single weight matrix.

    Exactly one of `sparsity` (unstructured) or `nm` (structured N:M) must be provided.

    Args:
        W: weight matrix (out, in).
        scaler: per-input-feature scaling vector (in,).
        sparsity: fraction to prune (0..1), e.g., 0.5.
        nm: (n, m) for structured N:M pruning.

    Returns:
        P: bool mask, True means keep.
    """
    if W.ndim != 2:
        raise ValueError(f"Expected 2D W, got {W.shape}")
    if scaler.ndim != 1 or scaler.shape[0] != W.shape[1]:
        raise ValueError(f"scaler must be shape (in_features,), got {scaler.shape} for W {W.shape}")

    score = W.abs() * scaler.to(device=W.device, dtype=W.dtype).unsqueeze(0)  # broadcast over out dim

    if (sparsity is None) == (nm is None):
        raise ValueError("Provide exactly one of sparsity or nm")

    if sparsity is not None:
        if not (0.0 <= sparsity < 1.0):
            raise ValueError(f"sparsity must be in [0,1), got {sparsity}")

        # Wanda's reference implementation performs *row-wise* pruning:
        # for each output row, prune the smallest `sparsity` fraction across input features.
        in_features = score.shape[1]
        k_prune = int(math.floor(in_features * sparsity))
        if k_prune <= 0:
            return torch.ones_like(W, dtype=torch.bool)
        if k_prune >= in_features:
            return torch.zeros_like(W, dtype=torch.bool)

        # Stable sort for deterministic behavior when there are ties.
        try:
            sort_res = torch.sort(score, dim=1, stable=True)
        except TypeError:
            # Older PyTorch versions may not support stable=True.
            sort_res = torch.sort(score, dim=1)
        indices = sort_res.indices[:, :k_prune]  # (out, k_prune) indices to prune
        prune_mask = torch.zeros_like(score, dtype=torch.bool)
        prune_mask.scatter_(1, indices, True)
        return ~prune_mask

    n, m = nm
    return nm_structured_mask(W, n=n, m=m, dim=1, score=score)
