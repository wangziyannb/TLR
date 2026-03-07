#!/usr/bin/env bash
set -euo pipefail

# Reproduce (approx.) Table 1 style WikiText-2 PPL sweeps for a 7B model.
# NOTE: This will be VERY compute intensive if you use --iters 50 on all layers.

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
OUT=${OUT:-runs/table1_llama2_7b}
SEQ_LEN=${SEQ_LEN:-2048}
WIKITEXT_SEQS=${WIKITEXT_SEQS:-128}
RANK=${RANK:-512}
ITERS=${ITERS:-50}

mkdir -p "$OUT"

# Unstructured sparsity levels
for SP in 0.5 0.6 0.7; do
  for REF in none zerosvd ours; do
    python apply_prune_refine.py \
      --model "$MODEL" \
      --pruning magnitude --sparsity "$SP" \
      --refine "$REF" --rank "$RANK" --iters "$ITERS" \
      --eval_ppl --seq_len "$SEQ_LEN" --wikitext_seqs "$WIKITEXT_SEQS" \
      --output_dir "$OUT/mag_sp${SP}_${REF}"
  done
done

# Structured patterns (4:8 and 2:4)
for NM in "4 8" "2 4"; do
  N=$(echo $NM | awk '{print $1}')
  M=$(echo $NM | awk '{print $2}')
  for REF in none zerosvd ours; do
    python apply_prune_refine.py \
      --model "$MODEL" \
      --pruning magnitude --nm $N $M \
      --refine "$REF" --rank "$RANK" --iters "$ITERS" \
      --eval_ppl --seq_len "$SEQ_LEN" --wikitext_seqs "$WIKITEXT_SEQS" \
      --output_dir "$OUT/mag_${N}of${M}_${REF}"
  done
done

echo "All runs completed. Results are in $OUT/*/results.json"
