#!/usr/bin/env bash
set -euo pipefail

# Wanda + refinement sweeps (subset of Table 1)
MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
OUT=${OUT:-runs/wanda_table1_llama2_7b}
SEQ_LEN=${SEQ_LEN:-2048}
WIKITEXT_SEQS=${WIKITEXT_SEQS:-128}
C4_SEQS=${C4_SEQS:-128}
RANK=${RANK:-32}
ITERS=${ITERS:-50}
LOAD_CALIBRATION_PATH=${LOAD_CALIBRATION_PATH:-/workspace/TLR/calib_llama2-7b.pth}

mkdir -p "$OUT"

#for SP in 0.5 0.6 0.7; do
#  for REF in none zerosvd ours; do
#    python apply_prune_refine.py \
#      --model "$MODEL" \
#      --pruning wanda --sparsity "$SP" \
#      --c4_seqs "$C4_SEQS" --seq_len "$SEQ_LEN" --use_c4_streaming \
#      --refine "$REF" --rank "$RANK" --iters "$ITERS" \
#      --eval_ppl --wikitext_seqs "$WIKITEXT_SEQS" \
#      --output_dir "$OUT/wanda_sp${SP}_${REF}_${RANK}"
#  done
#done
#
#for NM in "4 8" "2 4"; do
#  N=$(echo $NM | awk '{print $1}')
#  M=$(echo $NM | awk '{print $2}')
#  for REF in none zerosvd ours; do
#    python apply_prune_refine.py \
#      --model "$MODEL" \
#      --pruning wanda --nm $N $M \
#      --c4_seqs "$C4_SEQS" --seq_len "$SEQ_LEN" --use_c4_streaming \
#      --refine "$REF" --rank "$RANK" --iters "$ITERS" \
#      --eval_ppl --wikitext_seqs "$WIKITEXT_SEQS" \
#      --output_dir "$OUT/wanda_${N}of${M}_${REF}_${RANK}"
#  done
#done

#for SP in 0.5; do
#  for REF in ours; do
#    python apply_prune_refine.py \
#      --model "$MODEL" \
#      --pruning wanda --sparsity "$SP" \
#      --c4_seqs "$C4_SEQS" --seq_len "$SEQ_LEN" --use_c4_streaming \
#      --refine "$REF" --rank "$RANK" --iters "$ITERS" \
#      --eval_ppl --wikitext_seqs "$WIKITEXT_SEQS" \
#      --load_calibration_path "$LOAD_CALIBRATION_PATH" --export_param_dict \
#      --output_dir "$OUT/wanda_sp${SP}_${REF}_${RANK}"
#  done
#done

for NM in "2 4"; do
  N=$(echo $NM | awk '{print $1}')
  M=$(echo $NM | awk '{print $2}')
  for REF in ours; do
    python apply_prune_refine.py \
      --model "$MODEL" \
      --pruning wanda --nm $N $M \
      --c4_seqs "$C4_SEQS" --seq_len "$SEQ_LEN" --use_c4_streaming \
      --refine "$REF" --rank "$RANK" --iters "$ITERS" \
      --load_calibration_path "$LOAD_CALIBRATION_PATH" --export_param_dict \
      --output_dir "$OUT/wanda_${N}of${M}_${REF}_${RANK}"
  done
done

echo "All runs completed. Results are in $OUT/*/results.json"
