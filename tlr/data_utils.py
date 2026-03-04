"""Dataset utilities for calibration and evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def load_llama_tokenizer(model_name_or_path: str):
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    # Some LLaMA tokenizers have no pad token; for eval we can set it to eos.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def iter_c4_calibration_batches(
    tokenizer,
    *,
    seq_len: int = 2048,
    num_sequences: int = 128,
    seed: int = 0,
    streaming: bool = True,
) -> Iterator[Dict[str, torch.Tensor]]:
    """Yield tokenized calibration batches from C4.

    Uses streaming dataset by default to avoid downloading the whole corpus.
    Each yielded batch is of shape (1, seq_len).
    """
    ds = load_dataset("allenai/c4", "en", split="train", streaming=streaming)
    # Deterministic-ish sampling: we just take the first num_sequences long-enough examples.
    # For closer reproduction, you may implement random skip using seed.
    n = 0
    for ex in ds:
        text = ex.get("text", "")
        if not text or len(text) < 10:
            continue
        ids = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
        if ids.numel() < seq_len:
            continue
        ids = ids[:seq_len].unsqueeze(0)
        attn = torch.ones_like(ids)
        yield {"input_ids": ids, "attention_mask": attn}
        n += 1
        if n >= num_sequences:
            break


def get_wikitext2_eval_batches(
    tokenizer,
    *,
    seq_len: int = 2048,
    num_sequences: int = 128,
    split: str = "validation",
) -> List[Dict[str, torch.Tensor]]:
    """Prepare WikiText-2 evaluation batches for perplexity.

    This follows the common evaluation used in LLM pruning papers:
    - concatenate the entire split text
    - tokenize
    - chunk into non-overlapping blocks of seq_len
    - use first num_sequences blocks
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]
    # Drop last partial
    total_len = (input_ids.numel() // seq_len) * seq_len
    input_ids = input_ids[:total_len]
    input_ids = input_ids.view(-1, seq_len)
    if input_ids.shape[0] < num_sequences:
        raise ValueError(f"Not enough sequences: have {input_ids.shape[0]}, need {num_sequences}")
    input_ids = input_ids[:num_sequences]
    attn = torch.ones_like(input_ids)
    batches = []
    for i in range(num_sequences):
        batches.append({"input_ids": input_ids[i:i+1], "attention_mask": attn[i:i+1]})
    return batches
