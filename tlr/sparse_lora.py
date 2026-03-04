"""Sparse + low-rank (LoRA-style) linear wrapper.

The paper's final weight approximation per layer is:
    W ≈ S' + L_k
where S' preserves the pruning mask, and L_k is low-rank.

We implement:
    y = x @ S'^T + (x @ A^T) @ B^T
where L_k = B @ A and rank(B @ A) = k.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SparseLoRAState:
    weight_sparse: torch.Tensor  # (out, in)
    lora_B: torch.Tensor         # (out, k)
    lora_A: torch.Tensor         # (k, in)
    bias: Optional[torch.Tensor] = None


class SparseLoRALinear(nn.Module):
    """A drop-in replacement for nn.Linear with sparse base weights + low-rank patch."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)

        self.weight_sparse = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.lora_A = nn.Parameter(torch.empty((rank, in_features), device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty((out_features, rank), device=device, dtype=dtype))

        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,), device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        # init to zeros (we'll overwrite with computed values)
        with torch.no_grad():
            self.weight_sparse.zero_()
            self.lora_A.zero_()
            self.lora_B.zero_()
            if self.bias is not None:
                self.bias.zero_()

    @torch.no_grad()
    def load_state(self, state: SparseLoRAState) -> None:
        if state.weight_sparse.shape != self.weight_sparse.shape:
            raise ValueError(f"weight shape mismatch: {state.weight_sparse.shape} vs {self.weight_sparse.shape}")
        if state.lora_A.shape != self.lora_A.shape:
            raise ValueError(f"A shape mismatch: {state.lora_A.shape} vs {self.lora_A.shape}")
        if state.lora_B.shape != self.lora_B.shape:
            raise ValueError(f"B shape mismatch: {state.lora_B.shape} vs {self.lora_B.shape}")
        self.weight_sparse.copy_(state.weight_sparse)
        self.lora_A.copy_(state.lora_A)
        self.lora_B.copy_(state.lora_B)
        if self.bias is not None:
            if state.bias is None:
                raise ValueError("bias expected but state.bias is None")
            self.bias.copy_(state.bias)

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        weight_sparse: torch.Tensor,
        lora_B: torch.Tensor,
        lora_A: torch.Tensor,
    ) -> "SparseLoRALinear":
        """Build SparseLoRALinear from an existing nn.Linear."""
        if linear.weight.shape != weight_sparse.shape:
            raise ValueError("weight_sparse shape must match original linear.weight")
        out_features, in_features = linear.weight.shape
        rank = int(lora_A.shape[0])
        m = cls(in_features, out_features, rank, bias=(linear.bias is not None), device=linear.weight.device, dtype=linear.weight.dtype)
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        m.load_state(SparseLoRAState(weight_sparse=weight_sparse, lora_B=lora_B, lora_A=lora_A, bias=bias))
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base sparse linear
        out = F.linear(x, self.weight_sparse, self.bias)
        # Low-rank patch: x @ A^T -> (.., rank), then @ B^T -> (.., out_features)
        # F.linear uses weight.T internally.
        z = F.linear(x, self.lora_A, bias=None)
        out = out + F.linear(z, self.lora_B, bias=None)
        return out

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """Return dense merged weight W_hat = S + B@A (for debugging)."""
        return self.weight_sparse + self.lora_B @ self.lora_A

    @torch.no_grad()
    def nonzero_count(self) -> int:
        """Count non-zero parameters in the sparse matrix only."""
        return int((self.weight_sparse != 0).sum().item())
