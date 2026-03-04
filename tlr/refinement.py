"""Low-rank refinement methods.

Implements:
- Baseline 1: Zero-shot SVD refinement (paper Sec. 4, Baseline 1).
- Baseline 2: PCP with mask (paper Sec. 4, Baseline 2) [VERY slow; mainly for small matrices].
- Proposed method: Iterative weight update with adaptive rank increase (Algorithm 1).

Reference paper:
Targeted Low-rank Refinement: Enhancing Sparse Language Models with Precision
https://openreview.net/pdf?id=S0ncZdwcLt
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Tuple

import torch


@dataclass(frozen=True)
class SVDConfig:
    """Config for truncated / approximate SVD."""

    backend: Literal["lowrank", "full"] = "lowrank"
    oversample: int = 8
    niter: int = 2
    # If min(m, n) <= full_svd_dim_threshold, we will switch to full SVD automatically.
    full_svd_dim_threshold: int = 2048


@dataclass(frozen=True)
class PCPConfig:
    """Config for PCP-with-mask baseline (Baseline 2).

    NOTE:
        This baseline requires repeated SVDs and is extremely slow for large matrices.
        It is mainly useful to reproduce the paper's small-matrix diagnostics (e.g., Figure 2),
        or for small models/layers.
    """

    max_iter: int = 2000
    tol: float = 1e-6
    # Paper suggests λ = 1/sqrt(max(m,n)).
    lam: float | None = None
    # Augmented Lagrangian parameter (mu). If None, we set a heuristic based on matrix norm.
    mu: float | None = None
    # Hard safety guard: refuse to run if min(m,n) is larger than this, unless you override.
    max_dim: int = 2048
    verbose: bool = False


def _sorted_topk_from_lowrank(U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, k: int):
    # torch.svd_lowrank usually returns sorted singular values, but sort defensively.
    idx = torch.argsort(S, descending=True)
    U = U[:, idx]
    V = V[:, idx]
    S = S[idx]
    return U[:, :k], S[:k], V[:, :k]


@torch.no_grad()
def topk_svd(
    mat: torch.Tensor,
    k: int,
    *,
    cfg: SVDConfig = SVDConfig(),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute top-k SVD factors for `mat`.

    Returns (U_k, S_k, V_k) such that:
        mat ≈ U_k @ diag(S_k) @ V_k^T
    where U_k is (m, k), S_k is (k,), V_k is (n, k).
    """
    if mat.ndim != 2:
        raise ValueError(f"topk_svd expects a 2D tensor, got shape {tuple(mat.shape)}")
    m, n = mat.shape
    k_eff = min(k, m, n)
    if k_eff <= 0:
        raise ValueError(f"k must be positive, got k={k}")

    # Switch to full SVD for small matrices (faster and more accurate).
    if cfg.backend == "full" or min(m, n) <= cfg.full_svd_dim_threshold:
        # full_matrices=False gives reduced SVD.
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        # torch.linalg.svd returns S sorted desc.
        U_k = U[:, :k_eff]
        S_k = S[:k_eff]
        V_k = Vh.transpose(-2, -1)[:, :k_eff]  # V
        return U_k, S_k, V_k

    # Low-rank / randomized SVD.
    # torch.svd_lowrank expects floating type.
    if not mat.is_floating_point():
        mat = mat.float()

    q = min(k_eff + max(cfg.oversample, 0), min(m, n))
    # niter controls number of power iterations (accuracy vs speed)
    U, S, V = torch.svd_lowrank(mat, q=q, niter=max(cfg.niter, 0))
    return _sorted_topk_from_lowrank(U, S, V, k_eff)


@torch.no_grad()
def lowrank_reconstruct(U: torch.Tensor, S: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Reconstruct matrix from SVD factors.

    Args:
        U: (m, r)
        S: (r,)
        V: (n, r)

    Returns:
        (m, n) matrix U diag(S) V^T
    """
    if U.ndim != 2 or V.ndim != 2 or S.ndim != 1:
        raise ValueError("Expected U and V 2D, S 1D")
    if U.shape[1] != S.shape[0] or V.shape[1] != S.shape[0]:
        raise ValueError(f"Incompatible shapes: U={U.shape}, S={S.shape}, V={V.shape}")
    return (U * S.unsqueeze(0)) @ V.transpose(0, 1)


@torch.no_grad()
def compute_patch_factors(
    residual: torch.Tensor,
    k: int,
    *,
    cfg: SVDConfig = SVDConfig(),
    dtype: torch.dtype | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute low-rank patch factors (B, A) for residual.

    Returns B, A such that residual ≈ B @ A, with:
        B: (out_features, k)
        A: (k, in_features)
    """
    U_k, S_k, V_k = topk_svd(residual, k, cfg=cfg)
    B = U_k * S_k.unsqueeze(0)  # (m, k)
    A = V_k.transpose(0, 1)     # (k, n)
    if dtype is not None:
        B = B.to(dtype)
        A = A.to(dtype)
    return B, A


@torch.no_grad()
def zero_shot_svd_refine(
    W: torch.Tensor,
    P: torch.Tensor,
    *,
    k: int = 128,
    svd_cfg: SVDConfig = SVDConfig(),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Baseline 1: zero-shot SVD refinement with fixed sparse weights.

    Args:
        W: dense weight matrix (m, n)
        P: binary mask (same shape), 1 means kept (non-zero allowed).
        k: target rank for low-rank patch.

    Returns:
        S: sparse weight matrix (masked, same shape as W)
        B: (m, k)
        A: (k, n)
    """
    if W.shape != P.shape:
        raise ValueError(f"W and P must have same shape, got {W.shape} vs {P.shape}")
    W_dtype = W.dtype
    Wf = W.float()
    Pf = P.to(dtype=Wf.dtype)
    S = Wf * Pf
    residual = Wf - S
    B, A = compute_patch_factors(residual, k, cfg=svd_cfg, dtype=W_dtype)
    return S.to(W_dtype), B, A


@torch.no_grad()
def pcp_with_mask_refine(
    W: torch.Tensor,
    P: torch.Tensor,
    *,
    k: int = 128,
    pcp_cfg: PCPConfig = PCPConfig(),
    svd_cfg: SVDConfig = SVDConfig(backend="full"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Baseline 2: PCP with mask (Eq.(5)).

    This is a robust-PCA-style solver (inexact ALM / ADMM) with an additional projection
    that forces the sparse component to follow the fixed mask P.

    WARNING:
        This baseline is extremely slow for large matrices, because it runs many SVDs.

    Returns:
        S: learned sparse matrix (same mask P)
        B, A: low-rank patch factors from rank-k approximation of the learned low-rank matrix L
    """
    if W.shape != P.shape:
        raise ValueError(f"W and P must have same shape, got {W.shape} vs {P.shape}")
    if W.ndim != 2:
        raise ValueError(f"Expected 2D W, got {W.shape}")

    m, n = W.shape
    if min(m, n) > pcp_cfg.max_dim:
        raise ValueError(
            f"PCP baseline refused for shape {tuple(W.shape)} (min dim > {pcp_cfg.max_dim}). "
            "This baseline is meant for small matrices; use zerosvd/ours for LLM layers."
        )

    W_dtype = W.dtype
    Wf = W.float()
    Pf = P.to(dtype=Wf.dtype)

    lam = pcp_cfg.lam
    if lam is None:
        lam = 1.0 / math.sqrt(max(m, n))  # per paper
    mu = pcp_cfg.mu
    if mu is None:
        # heuristic: mu = 1.25 / ||W||_2 (common robust PCA choice)
        # ||W||_2 is spectral norm, compute via SVD for small matrices
        smax = torch.linalg.svdvals(Wf)[0]
        mu = float(1.25 / max(smax.item(), 1e-6))

    # Initialize
    L = torch.zeros_like(Wf)
    S = torch.zeros_like(Wf)
    Y = torch.zeros_like(Wf)

    normW = torch.linalg.norm(Wf, ord="fro")
    inv_mu = 1.0 / mu

    def svt(X: torch.Tensor, tau: float) -> torch.Tensor:
        # Singular value thresholding: U diag(max(s - tau, 0)) V^T
        U, s, Vh = torch.linalg.svd(X, full_matrices=False)
        s_thr = torch.clamp(s - tau, min=0.0)
        return (U * s_thr.unsqueeze(0)) @ Vh

    for it in range(pcp_cfg.max_iter):
        # L-update
        L = svt(Wf - S + inv_mu * Y, tau=inv_mu)

        # S-update with soft-thresholding, then project onto mask P
        X = Wf - L + inv_mu * Y
        S = torch.sign(X) * torch.clamp(X.abs() - lam * inv_mu, min=0.0)
        S = S * Pf

        # Dual update
        Z = Wf - L - S
        Y = Y + mu * Z

        err = torch.linalg.norm(Z, ord="fro") / (normW + 1e-12)
        if pcp_cfg.verbose and (it == 0 or (it + 1) % 50 == 0 or it == pcp_cfg.max_iter - 1):
            print(f"[PCP iter {it+1}/{pcp_cfg.max_iter}] rel_err={err.item():.3e}")
        if err.item() < pcp_cfg.tol:
            break

    # Low-rank patch: take best rank-k approximation of L
    B, A = compute_patch_factors(L, k, cfg=svd_cfg, dtype=W_dtype)
    return S.to(W_dtype), B, A


@torch.no_grad()
def iterative_weight_update_refine(
    W: torch.Tensor,
    P: torch.Tensor,
    *,
    k: int = 128,
    T: int = 50,
    r_start: int = 1,
    svd_cfg: SVDConfig = SVDConfig(),
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Proposed method: Iterative weight update refinement (Algorithm 1).

    Implements:
        Initialize S^(0) = W ⊙ P
        For t in [0, T-1]:
            L^(t) = W - S^(t)
            r^(t) = floor(r_start + (k - r_start) * t / (T - 1))
            S^(t+1) = S^(t) + P ⊙ (L^(t) - (L^(t))_r^(t))
        Final residual L^(T) = W - S^(T)
        Patch L_k is obtained via top-k SVD of L^(T)

    Notes:
        We compute the tail (from r^(t) onwards) as:
            L_tail = L - L_r
        where L_r is the best rank-r approximation (top-r singular components).
        This avoids needing the full SVD.

    Args:
        W: dense weight matrix (m, n)
        P: binary mask (same shape), 1 means kept.
        k: target rank of low-rank patch.
        T: number of iterations.
        r_start: starting retained rank in the schedule. Paper uses r(t)=floor(1 + (k-1)t/(T-1)).
        svd_cfg: config for truncated SVD used to build rank-r approximation.
        verbose: print per-iteration diagnostics (slow).

    Returns:
        S: refined sparse weight matrix
        B, A: low-rank patch factors with rank k
    """
    if W.shape != P.shape:
        raise ValueError(f"W and P must have same shape, got {W.shape} vs {P.shape}")
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if r_start < 1:
        raise ValueError(f"r_start must be >= 1, got {r_start}")

    W_dtype = W.dtype
    Wf = W.float()
    Pf = P.to(dtype=Wf.dtype)

    # Sparse term S^(0)
    S = Wf * Pf

    m, n = W.shape
    min_dim = min(m, n)

    def r_of_t(t: int) -> int:
        if T == 1:
            return min(k, min_dim)
        # Linear schedule: r_start -> k
        r = math.floor(r_start + (k - r_start) * (t / (T - 1)))
        r = max(1, min(r, k, min_dim))
        return int(r)

    for t in range(T):
        L = Wf - S
        r = r_of_t(t)

        # Compute best rank-r approximation of L: L_r
        if r >= min_dim:
            L_r = L
        else:
            U_r, S_r, V_r = topk_svd(L, r, cfg=svd_cfg)
            L_r = lowrank_reconstruct(U_r, S_r, V_r)

        # Tail from r onwards: L - L_r
        tail = L - L_r
        # Update only on the kept entries (mask P)
        S = S + Pf * tail
        # Enforce sparsity pattern exactly
        S = S * Pf

        if verbose and (t == 0 or (t + 1) % 5 == 0 or t == T - 1):
            # Diagnostic: Frobenius norm of error at target rank k
            U_k, S_k, V_k = topk_svd(Wf - S, min(k, min_dim), cfg=svd_cfg)
            approx = S + lowrank_reconstruct(U_k, S_k, V_k)
            err = torch.linalg.norm(Wf - approx, ord="fro")
            print(f"[iter {t+1:03d}/{T}] r={r}  err_fro={err.item():.4e}")

    residual = Wf - S
    B, A = compute_patch_factors(residual, k, cfg=svd_cfg, dtype=W_dtype)

    return S.to(W_dtype), B, A
