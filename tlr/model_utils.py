"""Helpers for working with HuggingFace transformer models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class NamedModule:
    name: str
    module: nn.Module
    parent: nn.Module
    attr: str


def _get_parent_and_attr(model: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    if len(parts) == 1:
        return model, parts[0]
    parent = model
    for p in parts[:-1]:
        if not hasattr(parent, p):
            raise AttributeError(f"Model has no submodule '{p}' in '{qualified_name}'")
        parent = getattr(parent, p)
    return parent, parts[-1]


def get_module(model: nn.Module, qualified_name: str) -> nn.Module:
    parent, attr = _get_parent_and_attr(model, qualified_name)
    return getattr(parent, attr)


def set_module(model: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    parent, attr = _get_parent_and_attr(model, qualified_name)
    setattr(parent, attr, new_module)


def iter_named_linears(
    model: nn.Module,
    *,
    name_filter: Optional[Callable[[str, nn.Module], bool]] = None,
) -> Iterator[NamedModule]:
    """Yield (name, linear, parent, attr) for nn.Linear modules matching filter."""
    # Build dict of name->module to find parent quickly
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name_filter is not None and not name_filter(name, module):
            continue
        parent, attr = _get_parent_and_attr(model, name)
        yield NamedModule(name=name, module=module, parent=parent, attr=attr)


def llama_default_linear_filter(name: str, module: nn.Module) -> bool:
    """Default filter matching the 7 matrices per transformer block (attention + MLP).

    Excludes:
        - embed_tokens
        - lm_head
    """
    # Common HF LLaMA names:
    # model.embed_tokens
    # model.layers.N.self_attn.{q_proj,k_proj,v_proj,o_proj}
    # model.layers.N.mlp.{gate_proj,up_proj,down_proj}
    # lm_head
    if "embed_tokens" in name:
        return False
    if name.endswith("lm_head") or name.startswith("lm_head"):
        return False
    if ".layers." not in name:
        return False
    if ".self_attn." in name or ".mlp." in name:
        return True
    return False


@torch.no_grad()
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_nonzero_parameters(model: nn.Module) -> int:
    """Counts non-zeros across all parameters (dense view).

    Note: if you use SparseLoRALinear, this will count lora params too.
    """
    total = 0
    for p in model.parameters():
        if p.is_floating_point():
            total += int((p != 0).sum().item())
        else:
            total += p.numel()
    return total
