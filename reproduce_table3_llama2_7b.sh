#!/usr/bin/env bash
set -euo pipefail

# Approximate reproduction of Table 3 for 7B model:
# 1) build pruned+refined model and export merged HF model
# 2) run lm_eval tasks (0-shot)

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
OUT=${OUT:-runs/table3_llama2_7b_mag50_ours}
SEQ_LEN=${SEQ_LEN:-2048}
RANK=${RANK:-128}
ITERS=${ITERS:-50}
DEVICE=${DEVICE:-cuda:0}

python scripts/apply_prune_refine.py \
  --model "$MODEL" \
  --pruning magnitude --sparsity 0.5 \
  --refine ours --rank "$RANK" --iters "$ITERS" \
  --output_dir "$OUT" \
  --export_merged_hf

# lm-eval-harness
lm_eval \
  --model hf \
  --model_args pretrained=$OUT/hf_merged_model,tokenizer=$OUT/hf_merged_model \
  --tasks truthfulqa_mc2,gsm8k,arc_challenge,mmlu \
  --num_fewshot 0 \
  --batch_size 1 \
  --device "$DEVICE"
