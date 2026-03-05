# Unofficial Targeted Low-rank Refinement (TLR)

这是一个从论文描述 **重新实现** 的 codebase，用于复现 OpenReview 论文：

**Targeted Low-rank Refinement: Enhancing Sparse Language Models with Precision**  
OpenReview PDF: https://openreview.net/pdf?id=S0ncZdwcLt

论文核心思想：对每个被剪枝的线性层权重矩阵 `W`（形状 `m×n`），在保持剪枝 mask `P` 不变的前提下，
学习一个“更好的稀疏矩阵” `S'`，并补一个 rank-k 的低秩 patch `L_k`，使得：

	`W ≈ S' + L_k`

其中 `S' = S' ⊙ P`，保证稀疏结构不变；`L_k` 用 SVD 得到并用 (B@A) 存储，参数量是 `k(m+n)`。

本仓库实现了：
- **Baseline 1：Zero-shot SVD**（固定 `S=W⊙P`，对残差 `W-S` 做 top-k SVD）
- **Ours：Iterative Weight Update**（论文 Algorithm 1，T=50，k=128 默认）

以及生成剪枝 mask 的：
- **Magnitude pruning**（unstructured + N:M 结构化）
- **Wanda pruning**（需要 C4 calibration；论文用 128 sequences）

---

## 1. 安装

```bash
git clone <this_repo>
cd tlr_repro
pip install -r requirements.txt
```

你需要具备可用的 LLaMA 权重（例如 HuggingFace 上的 `meta-llama/Llama-2-7b-hf`），并已完成授权。

---

## 2. 关键脚本：剪枝 + 低秩修复 + WikiText-2 PPL

下面示例默认复现论文中常用设置：
- `k=128`
- `T=50`
- WikiText-2 perplexity 用 `128` 个 `seq_len=2048` 的 chunk

### 2.1 Magnitude pruning (50% unstructured) + Ours

```bash
python scripts/apply_prune_refine.py       --model meta-llama/Llama-2-7b-hf       --pruning magnitude --sparsity 0.5       --refine ours --rank 128 --iters 50       --eval_ppl --seq_len 2048 --wikitext_seqs 128       --output_dir runs/llama2_7b_mag50_ours
```

### 2.2 Magnitude pruning (50%) + Zero-shot SVD

```bash
python scripts/apply_prune_refine.py       --model meta-llama/Llama-2-7b-hf       --pruning magnitude --sparsity 0.5       --refine zerosvd --rank 128       --eval_ppl --seq_len 2048 --wikitext_seqs 128       --output_dir runs/llama2_7b_mag50_zerosvd
```

### 2.3 Wanda pruning + Ours（需要 C4 calibration）

```bash
python scripts/apply_prune_refine.py       --model meta-llama/Llama-2-7b-hf       --pruning wanda --sparsity 0.5       --c4_seqs 128 --seq_len 2048 --use_c4_streaming       --refine ours --rank 128 --iters 50       --eval_ppl --wikitext_seqs 128       --output_dir runs/llama2_7b_wanda50_ours
```

### 2.4 结构化 N:M（例如 2:4）

```bash
python scripts/apply_prune_refine.py       --model meta-llama/Llama-2-7b-hf       --pruning magnitude --nm 2 4       --refine ours --rank 128 --iters 50       --eval_ppl --seq_len 2048 --wikitext_seqs 128       --output_dir runs/llama2_7b_mag_2of4_ours
```

---

## 3. 复现 Table 3 的 benchmark（TruthfulQA / GSM8K / ARC-c / MMLU）

论文 Table 3 多数数值看起来是 **0-shot**（建议在 lm-eval-harness 里用 `--num_fewshot 0`）。
为了让 `lm_eval` 直接读取模型，本仓库提供 `--export_merged_hf`：
- 会把每层 `S + (B@A)` 合并回 `nn.Linear.weight`（变成 dense 权重）
- 然后 `save_pretrained()` 到 `runs/.../hf_merged_model`

**导出 HF 模型：**
```bash
python scripts/apply_prune_refine.py       --model meta-llama/Llama-2-7b-hf       --pruning magnitude --sparsity 0.5       --refine ours --rank 128 --iters 50       --output_dir runs/llama2_7b_mag50_ours       --export_merged_hf
```

**用 lm-eval-harness 跑 Table 3 的任务：**
```bash
lm_eval       --model hf       --model_args pretrained=runs/llama2_7b_mag50_ours/hf_merged_model,tokenizer=runs/llama2_7b_mag50_ours/hf_merged_model       --tasks truthfulqa_mc2,gsm8k,arc_challenge,mmlu       --num_fewshot 0       --batch_size 1       --device cuda:0
```

你可以把输出的四个任务 accuracy 做平均，得到和论文 Table 3 一样的 AVG。

---

## 4. 代码导读（对应论文）

- `tlr/refinement.py`
  - `zero_shot_svd_refine`：Baseline 1
  - `iterative_weight_update_refine`：Algorithm 1（Eq.(7) 的更新）
- `tlr/sparse_lora.py`
  - `SparseLoRALinear`：把 `S` + `B@A` 作为前向计算
- `tlr/pruning/magnitude.py`
  - magnitude 与 N:M mask
- `tlr/pruning/wanda.py`
  - Wanda score：`|W| * sqrt(mean(x^2))`
- `scripts/apply_prune_refine.py`
  - 端到端：load model → mask → refine → ppl → export

---
