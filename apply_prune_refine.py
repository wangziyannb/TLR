#!/usr/bin/env python
"""Prune + (optional) low-rank refinement for LLaMA-family models.

This script is the main entry point for reproducing the paper's results.

Typical usage (WikiText-2 perplexity):
  python scripts/apply_prune_refine.py \
    --model meta-llama/Llama-2-7b-hf \
    --pruning magnitude --sparsity 0.5 \
    --refine ours --rank 128 --iters 50 \
    --eval_ppl --seq_len 2048 --wikitext_seqs 128 \
    --output_dir runs/llama2_7b_mag50_ours

Wanda pruning (requires calibration):
  python scripts/apply_prune_refine.py \
    --model meta-llama/Llama-2-7b-hf \
    --pruning wanda --sparsity 0.5 \
    --c4_seqs 128 --seq_len 2048 \
    --refine ours --rank 128 --iters 50 \
    --eval_ppl --wikitext_seqs 128 \
    --output_dir runs/llama2_7b_wanda50_ours

Structured N:M (e.g., 2:4):
  python scripts/apply_prune_refine.py \
    --model meta-llama/Llama-2-7b-hf \
    --pruning magnitude --nm 2 4 \
    --refine ours --rank 128 --iters 50 \
    --eval_ppl --output_dir runs/llama2_7b_mag2of4_ours
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from tlr.data_utils import (
    get_wikitext2_eval_batches,
    iter_c4_calibration_batches,
    load_llama_tokenizer,
)
from tlr.eval.ppl import eval_ppl
from tlr.model_utils import (
    iter_named_linears,
    llama_default_linear_filter,
    set_module,
)
from tlr.pruning import magnitude_mask, nm_structured_mask
from tlr.pruning.wanda import collect_wanda_stats, wanda_mask
from tlr.refinement import SVDConfig, PCPConfig, iterative_weight_update_refine, zero_shot_svd_refine, pcp_with_mask_refine
from tlr.sparse_lora import SparseLoRALinear


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="HF model name or local path (e.g., meta-llama/Llama-2-7b-hf)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=0)

    # Pruning
    p.add_argument("--pruning", type=str, required=True, choices=["magnitude", "wanda"], help="Pruning method to create mask P")
    p.add_argument("--sparsity", type=float, default=None, help="Unstructured sparsity ratio p in [0,1). e.g., 0.5")
    p.add_argument("--nm", type=int, nargs=2, default=None, metavar=("N", "M"), help="Structured N:M pattern (e.g., --nm 2 4 or --nm 4 8)")

    # Wanda calibration
    p.add_argument("--c4_seqs", type=int, default=128, help="Number of C4 sequences for Wanda calibration (paper uses 128)")
    p.add_argument("--use_c4_streaming", action="store_true", help="Use streaming mode for C4 (recommended)")
    p.add_argument("--seq_len", type=int, default=2048)

    # Wanda mask generation mode
    p.add_argument(
        "--wanda_mode",
        type=str,
        default="sequential",
        choices=["sequential", "oneshot"],
        help=(
            "How to generate Wanda masks. 'sequential' matches the official WANDA repo (updates activations layer-by-layer). "
            "'oneshot' collects stats on the dense model in one pass (faster, but may deviate from reference results)."
        ),
    )
    p.add_argument(
        "--wanda_store_inps",
        type=str,
        default="gpu",
        choices=["cpu", "gpu"],
        help="Where to store calibration hidden states for sequential Wanda. cpu saves GPU memory.",
    )

    # Refinement
    p.add_argument("--refine", type=str, required=True, choices=["none", "zerosvd", "ours", "pcp"], help="Low-rank refinement method")
    p.add_argument("--rank", type=int, default=128, help="Target rank k (paper uses 128)")
    p.add_argument("--iters", type=int, default=50, help="Iterations T (paper uses 50)")
    p.add_argument("--svd_backend", type=str, default="lowrank", choices=["lowrank", "full"], help="SVD backend")
    p.add_argument("--svd_niter", type=int, default=2, help="Power iterations for randomized SVD (torch.svd_lowrank)")
    p.add_argument("--svd_oversample", type=int, default=8, help="Oversampling for randomized SVD")

    # PCP baseline (very slow; only for small matrices)
    p.add_argument("--pcp_iters", type=int, default=2000, help="Max iters for PCP-with-mask baseline (Baseline 2)")
    p.add_argument("--pcp_tol", type=float, default=1e-6, help="Tolerance for PCP-with-mask baseline")
    p.add_argument("--pcp_max_dim", type=int, default=2048, help="Refuse PCP if min(m,n) is larger than this")

    # Eval
    p.add_argument("--eval_ppl", action="store_true", help="Evaluate WikiText-2 perplexity")
    p.add_argument("--wikitext_seqs", type=int, default=128, help="Number of WikiText-2 sequences for perplexity (paper uses 128)")

    # Export
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--export_merged_hf", action="store_true", help="Export merged dense HF model (so you can run lm_eval easily)")
    p.add_argument("--max_layers", type=int, default=None, help="Debug: only process first N linear layers")

    return p.parse_args()


def str_to_dtype(s: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]


@torch.no_grad()
def merge_and_restore_linear(model: nn.Module) -> None:
    """Replace SparseLoRALinear modules with nn.Linear using merged weights."""
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, SparseLoRALinear):
            to_replace.append(name)
    for name in to_replace:
        m: SparseLoRALinear = getattr(model, name.split(".")[-1], None)  # not used, we use set_module below
        mod = None
        # Fetch actual module via traversal
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        mod = getattr(parent, parts[-1])
        assert isinstance(mod, SparseLoRALinear)

        lin = nn.Linear(mod.in_features, mod.out_features, bias=(mod.bias is not None), device=mod.weight_sparse.device, dtype=mod.weight_sparse.dtype)
        lin.weight.copy_(mod.merged_weight())
        if mod.bias is not None:
            lin.bias.copy_(mod.bias)
        set_module(model, name, lin)



@torch.no_grad()
def prepare_wanda_calibration_inputs(model, calib_batches, device: torch.device, dtype: torch.dtype, store_inps: str = "cpu"):
    """Prepare hidden-state inputs for layer-wise (sequential) Wanda pruning.

    This follows the official WANDA codepath: we capture the hidden-states that are fed into
    the first decoder layer, along with the internal attention_mask / position_ids tensors
    that HuggingFace passes to decoder layers.

    Args:
        model: HF causal LM (LLaMA-family assumed).
        calib_batches: list of tokenized batches, each (1, seq_len).
        device: compute device.
        dtype: hidden-state dtype to store (float16/bfloat16/float32).
        store_inps: "cpu" or "gpu" – where to store the large (nsamples, seq_len, hidden) tensors.

    Returns:
        inps, outs: tensors shaped (nsamples, seq_len, hidden_size) on store device.
        attention_mask, position_ids: tensors on compute device suitable for calling a single decoder layer.
    """
    model.eval()
    use_cache = getattr(model.config, "use_cache", False)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # LLaMA-style layout: model.model.layers is a ModuleList of decoder layers.
    try:
        layers = model.model.layers
    except AttributeError:
        try:
            layers = model.base_model.layers
        except AttributeError:
            layers = model.base_model.decoder.layers

    if len(layers) == 0:
        raise ValueError("Model has no decoder layers at model.model.layers")

    nsamples = len(calib_batches)
    if nsamples == 0:
        raise ValueError("calib_batches is empty")
    seqlen = int(calib_batches[0]["input_ids"].shape[1])
    hidden_size = int(model.config.hidden_size)

    store_device = device if store_inps == "gpu" else torch.device("cpu")
    inps = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype, device=store_device)
    outs = torch.zeros_like(inps)

    cache = {'i': 0, 'attention_mask': None, "position_ids": None, "cache_position": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module
            if hasattr(module, "self_attn"):
                self.self_attn = module.self_attn
            elif hasattr(module, "attn"):
                self.attn = module.attn
            if hasattr(module, "attention_type"):
                self.attention_type = self.module.attention_type

        def forward(self, inp, **kwargs):
            i = cache["i"]
            if i < nsamples:
                x = inp.detach()
                if x.dim() == 3:  # [B, T, H]
                    B = x.size(0)
                    for b in range(B):
                        if i >= inps.size(0):
                            break
                        inps[i].copy_(x[b].to(device=store_device, dtype=dtype))
                        i += 1
                else:  # [T, H]
                    inps[i].copy_(x.to(device=store_device, dtype=dtype))
                    i += 1
                # inps[i].copy_(inp.detach().to(device=store_device, dtype=dtype))
                cache["i"] = i
                # HF passes these to each decoder layer; needed when we call layers directly.
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs['position_ids']
                if 'cache_position' in kwargs and 'position_embeddings' in kwargs:
                    cache['cache_position'] = kwargs['cache_position']
                    cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError

    # Swap in catcher
    layers[0] = Catcher(layers[0])

    for batch in calib_batches:
        if cache["i"] >= nsamples:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        try:
            model(**batch, use_cache=False)
        except ValueError:
            pass

    # Restore first layer
    layers[0] = layers[0].module

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = use_cache

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    # if attention_mask is None or position_ids is None:
    #     # Fallback: works for some models, but for LLaMA this should not happen.
    #     attention_mask = torch.ones((1, seqlen), device=device, dtype=torch.long)
    #     position_ids = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(0)
    if 'cache_position' in cache and 'position_embeddings' in cache:
        cache_position = cache['cache_position']
        position_embeddings = cache['position_embeddings']
        return inps, outs, attention_mask, position_ids, cache_position, position_embeddings

    return inps, outs, attention_mask, position_ids


@torch.no_grad()
def apply_wanda_sequential_prune_and_refine(
    model: nn.Module,
    tok,
    args,
    *,
    device: torch.device,
    dtype: torch.dtype,
    nm_pattern,
    svd_cfg: SVDConfig,
) -> None:
    """Sequential (layer-wise) Wanda pruning, matching the official WANDA repo.

    Key differences vs the fast 'oneshot' mode:
      1) Activation statistics are collected *layer by layer*.
      2) After pruning each transformer block, we recompute and store its outputs as the
         inputs to the next block ("updated activations").

    After masks are computed, we optionally apply the paper's low-rank refinement per Linear.
    """
    print("Preparing C4 calibration batches for sequential Wanda...")
    calib_batches = list(
        iter_c4_calibration_batches(
            tok,
            seq_len=args.seq_len,
            num_sequences=args.c4_seqs,
            streaming=args.use_c4_streaming,
            seed=args.seed,
        )
    )
    if len(calib_batches) != args.c4_seqs:
        print(f"[warn] Requested {args.c4_seqs} calibration seqs but got {len(calib_batches)} from C4 iterator.")

    print("Capturing layer-0 inputs (WANDA-style)...")
    inps, outs, attention_mask, position_ids, cache_position, position_embeddings = prepare_wanda_calibration_inputs(
        model,
        calib_batches,
        device=device,
        dtype=dtype,
        store_inps=args.wanda_store_inps,
    )

    layers = model.model.layers
    amp_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float16

    processed_linears = 0
    total_layers = len(layers)

    for i, layer in enumerate(layers):
        # Find target Linear modules inside this transformer layer.
        subset = {}
        for rel_name, mod in layer.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            full_name = f"model.layers.{i}.{rel_name}"
            if llama_default_linear_filter(full_name, mod):
                subset[full_name] = mod

        if len(subset) == 0:
            # Still need to propagate activations forward.
            for j in range(len(calib_batches)):
                inp = inps[j].unsqueeze(0)
                if inps.device.type == "cpu":
                    inp = inp.to(device)
                with torch.autocast(device_type=str(device).split(":")[0], dtype=amp_dtype, enabled=(device.type == "cuda")):
                    out = layer(inp, attention_mask=attention_mask, position_ids=position_ids,cache_position= cache_position, position_embeddings=position_embeddings)[0]
                outs[j].copy_(out.detach().to(device=outs.device, dtype=outs.dtype))
            inps, outs = outs, inps
            continue

        # Accumulate activation stats for Wanda: s_j = sqrt(mean(x_j^2)).
        sum_sq = {name: torch.zeros((lin.in_features,), dtype=torch.float64, device="cpu") for name, lin in subset.items()}
        count = {name: 0 for name in subset.keys()}

        handles = []

        def make_hook(name: str):
            def hook(module: nn.Module, inputs, output):
                x = inputs[0]
                if not torch.is_tensor(x):
                    return
                x = x.detach()
                x = x.reshape(-1, x.shape[-1]).to(dtype=torch.float32)
                sum_sq[name] += (x * x).sum(dim=0).to(dtype=torch.float64, device="cpu")
                count[name] += x.shape[0]
            return hook

        for name, lin in subset.items():
            handles.append(lin.register_forward_hook(make_hook(name)))

        # Pass 1: run the (current) dense transformer layer to collect stats.
        for j in range(len(calib_batches)):
            inp = inps[j].unsqueeze(0)
            if inps.device.type == "cpu":
                inp = inp.to(device)
            with torch.autocast(device_type=str(device).split(":")[0], dtype=amp_dtype, enabled=(device.type == "cuda")):
                out = layer(
                    inp,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position= cache_position, position_embeddings=position_embeddings
                )[0]
        for h in handles:
            h.remove()

        # Build masks and prune weights in-place (and stash dense weights for refinement).
        masks = {}
        dense_W = {}

        for name, lin in subset.items():
            W = lin.weight.data
            if args.refine != "none":
                dense_W[name] = W.clone()
            scaler = torch.sqrt(sum_sq[name] / float(count[name])).to(device=W.device, dtype=W.dtype)

            if nm_pattern is None:
                P = wanda_mask(W, scaler, sparsity=args.sparsity)
            else:
                P = wanda_mask(W, scaler, nm=nm_pattern)

            masks[name] = P
            lin.weight.data = W * P.to(dtype=W.dtype)

        # Pass 2: recompute layer outputs with pruned weights to get updated activations.
        for j in range(len(calib_batches)):
            inp = inps[j].unsqueeze(0)
            if inps.device.type == "cpu":
                inp = inp.to(device)
            with torch.autocast(device_type=str(device).split(":")[0], dtype=amp_dtype, enabled=(device.type == "cuda")):
                out = layer(inp, attention_mask=attention_mask, position_ids=position_ids,cache_position= cache_position, position_embeddings=position_embeddings)[0]
            outs[j].copy_(out.detach().to(device=outs.device, dtype=outs.dtype))

        # Swap buffers so next layer sees pruned activations.
        inps, outs = outs, inps

        # Low-rank refinement (post-processing) on this layer's Linear modules.
        if args.refine != "none":
            for name, lin in subset.items():
                W_dense = dense_W[name]
                P = masks[name]
                if args.refine == "zerosvd":
                    S, B, A = zero_shot_svd_refine(W_dense, P, k=args.rank, svd_cfg=svd_cfg)
                elif args.refine == "ours":
                    S, B, A = iterative_weight_update_refine(W_dense, P, k=args.rank, T=args.iters, svd_cfg=svd_cfg)
                elif args.refine == "pcp":
                    pcp_cfg = PCPConfig(max_iter=args.pcp_iters, tol=args.pcp_tol, max_dim=args.pcp_max_dim)
                    S, B, A = pcp_with_mask_refine(W_dense, P, k=args.rank, pcp_cfg=pcp_cfg)
                else:
                    raise ValueError(args.refine)

                wrapped = SparseLoRALinear.from_linear(lin, weight_sparse=S, lora_B=B, lora_A=A)
                set_module(model, name, wrapped)

        processed_linears += len(subset)
        print(f"  Wanda-seq: finished transformer layer {i+1}/{total_layers} (linears processed: {processed_linears})")

        if args.max_layers is not None and processed_linears >= args.max_layers:
            print(f"[debug] --max_layers={args.max_layers}: stopping early.")
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = str_to_dtype(args.dtype)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load model + tokenizer
    tok = load_llama_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    # Collect candidate linear layers
    linears = list(iter_named_linears(model, name_filter=llama_default_linear_filter))
    if args.max_layers is not None:
        linears = linears[: args.max_layers]
    name_to_linear: Dict[str, nn.Linear] = {nm.name: nm.module for nm in linears}

    if args.nm is None and args.sparsity is None:
        raise SystemExit("You must provide either --sparsity (unstructured) or --nm N M (structured)")

    if args.nm is not None and args.sparsity is not None:
        raise SystemExit("Provide only one of --sparsity or --nm")

    nm_pattern = tuple(args.nm) if args.nm is not None else None

    # Wanda stats if needed
    wanda_stats = None
    if args.pruning == "wanda" and args.wanda_mode == "oneshot":
        print("[1/3] Collecting Wanda calibration statistics...")
        calib_iter = iter_c4_calibration_batches(
            tok,
            seq_len=args.seq_len,
            num_sequences=args.c4_seqs,
            streaming=args.use_c4_streaming,
            seed=args.seed,
        )
        wanda_stats = collect_wanda_stats(
            model,
            name_to_linear,
            calib_iter,
            device=device,
            amp_dtype=dtype if dtype in (torch.float16, torch.bfloat16) else torch.float16,
            use_cache=False,
        )
        print("Wanda stats collected.")

    # SVD config
    svd_cfg = SVDConfig(backend=args.svd_backend, oversample=args.svd_oversample, niter=args.svd_niter)

    # Apply pruning + refinement
    print("[2/3] Applying pruning + refinement...")

    if args.pruning == "wanda" and args.wanda_mode == "sequential":
        # Faithful Wanda implementation (layer-wise, with updated activations)
        apply_wanda_sequential_prune_and_refine(
            model,
            tok,
            args,
            device=device,
            dtype=dtype,
            nm_pattern=nm_pattern,
            svd_cfg=svd_cfg,
        )
    else:
        # Fast path: one-shot pruning masks, then independent refinement per Linear
        processed = 0
        for nm in linears:
            name = nm.name
            lin: nn.Linear = nm.module
            W = lin.weight.data

            # Build mask P
            if args.pruning == "magnitude":
                if nm_pattern is None:
                    P = magnitude_mask(W, args.sparsity)
                else:
                    n, m = nm_pattern
                    P = nm_structured_mask(W, n=n, m=m, dim=1)
            elif args.pruning == "wanda":
                assert wanda_stats is not None
                scaler = wanda_stats.scaler(name)
                if nm_pattern is None:
                    P = wanda_mask(W, scaler, sparsity=args.sparsity)
                else:
                    P = wanda_mask(W, scaler, nm=nm_pattern)
            else:
                raise ValueError(args.pruning)

            # Apply refinement
            if args.refine == "none":
                lin.weight.data = (W * P.to(dtype=W.dtype))
            else:
                if args.refine == "zerosvd":
                    S, B, A = zero_shot_svd_refine(W, P, k=args.rank, svd_cfg=svd_cfg)
                elif args.refine == "ours":
                    S, B, A = iterative_weight_update_refine(W, P, k=args.rank, T=args.iters, svd_cfg=svd_cfg)
                elif args.refine == "pcp":
                    pcp_cfg = PCPConfig(max_iter=args.pcp_iters, tol=args.pcp_tol, max_dim=args.pcp_max_dim)
                    S, B, A = pcp_with_mask_refine(W, P, k=args.rank, pcp_cfg=pcp_cfg)
                else:
                    raise ValueError(args.refine)

                wrapped = SparseLoRALinear.from_linear(lin, weight_sparse=S, lora_B=B, lora_A=A)
                set_module(model, name, wrapped)

            processed += 1
            if processed % 10 == 0 or processed == len(linears):
                print(f"  processed {processed}/{len(linears)} layers")

            if device.type == "cuda":
                torch.cuda.empty_cache()

    results = {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "pruning": args.pruning,
        "wanda_mode": args.wanda_mode,
        "wanda_store_inps": args.wanda_store_inps,
        "sparsity": args.sparsity,
        "nm": nm_pattern,
        "refine": args.refine,
        "rank": args.rank,
        "iters": args.iters,
    }

    # Evaluate perplexity
    if args.eval_ppl:
        print("[3/3] Evaluating WikiText-2 perplexity...")
        batches = get_wikitext2_eval_batches(tok, seq_len=args.seq_len, num_sequences=args.wikitext_seqs, split="validation")
        avg_loss, ppl = eval_ppl(model, batches, device=device, amp_dtype=dtype if dtype in (torch.float16, torch.bfloat16) else torch.float16)
        print(f"WikiText-2 loss={avg_loss:.4f}  ppl={ppl:.4f}")
        results["wikitext2_loss"] = avg_loss
        results["wikitext2_ppl"] = ppl

    # Save results json
    (outdir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Export merged HF model (optional)
    if args.export_merged_hf:
        print("Exporting merged dense HF model...")
        merge_and_restore_linear(model)
        export_dir = outdir / "hf_merged_model"
        export_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(export_dir)
        tok.save_pretrained(export_dir)
        print(f"Saved to: {export_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
