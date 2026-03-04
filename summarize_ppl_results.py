#!/usr/bin/env python
"""Summarize results.json files (perplexity) under a runs directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", type=str, help="Directory containing subfolders with results.json")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = []
    for p in sorted(runs_dir.glob("**/results.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        ppl = data.get("wikitext2_ppl", None)
        if ppl is None:
            continue
        rows.append((str(p.parent), data.get("pruning"), data.get("sparsity"), data.get("nm"), data.get("refine"), ppl))

    if not rows:
        print("No results found.")
        return

    # Pretty print
    print(f"Found {len(rows)} runs:\n")
    print(f"{'run':60s}  {'prune':9s}  {'sp':5s}  {'nm':7s}  {'ref':7s}  ppl")
    print("-" * 110)
    for r in rows:
        run, prune, sp, nm, ref, ppl = r
        nm_s = "-" if nm is None else f"{nm[0]}:{nm[1]}"
        sp_s = "-" if sp is None else f"{sp:.2f}"
        print(f"{run[:60]:60s}  {prune:9s}  {sp_s:5s}  {nm_s:7s}  {ref:7s}  {ppl:.4f}")

if __name__ == "__main__":
    main()
