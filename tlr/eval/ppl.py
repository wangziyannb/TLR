"""Perplexity evaluation for causal LMs on WikiText-2."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn


@torch.no_grad()
def eval_ppl(
    model: nn.Module,
    batches: Iterable[Dict[str, torch.Tensor]],
    *,
    device: torch.device,
    amp_dtype: torch.dtype = torch.float16,
) -> Tuple[float, float]:
    """Evaluate perplexity.

    Returns:
        (avg_loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    use_amp = device.type == "cuda"
    device_type = "cuda" if device.type == "cuda" else "cpu"

    for batch in batches:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # For HF causal LMs, passing labels computes next-token loss internally.
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = out.loss  # mean over tokens

        # Count tokens contributing to loss. For causal LM it's (seq_len - 1) per sequence if all unmasked.
        seq_len = input_ids.shape[1]
        batch_tokens = input_ids.shape[0] * (seq_len - 1)
        total_loss += float(loss.item()) * batch_tokens
        total_tokens += batch_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl
