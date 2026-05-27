# Encoder-Decoder SFT and Pruning

This repository provides a small, practical toolkit for:

1. Supervised fine-tuning from prompt-response data.
2. Exact-match generation evaluation against an eval file.
3. Vanilla pruning methods: magnitude, gradient, NVIDIA 2:4, and WANDA.

It supports both decoder-only checkpoints and encoder-decoder checkpoints such as `charent/ChatLM-mini-Chinese` or T5-style 0.5B models. The code auto-detects model family from the Hugging Face config, or you can force `--model_family causal` or `--model_family seq2seq`.

## Setup

Conda GPU environment:

```bash
conda env create -f environment.yml
conda activate encoder-decoder-prune
pip install -e ".[dev]"
```

Lightweight virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

See [docs/decoder_only_pruning_prompt.md](docs/decoder_only_pruning_prompt.md) for a copy-paste prompt that describes the expected decoder-only 50% pruning behavior against the CMC scripts.

## Data Format

Local `.jsonl`, `.json`, `.csv`, `.tsv`, Hugging Face datasets, and `datasets.save_to_disk(...)` directories are supported. The default columns are `prompt` and `response`.

```json
{"prompt":"Summarize: Transformers are sequence models.","response":"Transformers model sequences with attention."}
```

Use `--prompt_field` and `--response_field` if your columns differ.

## SFT

Decoder-only example:

```bash
encdec-sft \
  --model_name_or_path charent/ChatLM-mini-Chinese \
  --train_source examples/train.jsonl \
  --eval_source examples/eval.jsonl \
  --output_dir runs/chatlm-mini-sft \
  --model_family causal \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3
```

T5-style encoder-decoder example:

```bash
encdec-sft \
  --model_name_or_path path-or-hf-id-for-your-t5-0.5b \
  --train_source examples/train.jsonl \
  --eval_source examples/eval.jsonl \
  --output_dir runs/t5-sft \
  --model_family seq2seq \
  --source_max_length 768 \
  --target_max_length 256
```

For decoder-only models, only response tokens are used as labels. Prompt tokens are masked with `-100`, which matches the usual SFT behavior for causal LMs. You can set the prompt template used by causal models:

```bash
--causal_prompt_template "User: {prompt}\nAssistant: "
```

Use the same template for SFT and exact evaluation.

To train regular SFT and report exact-match plus top-5 generation accuracy on both the SFT dataset and a benchmark dataset:

```bash
encdec-sft \
  --model_name_or_path charent/ChatLM-mini-Chinese \
  --train_source data/sft.jsonl \
  --sft_eval_source data/sft.jsonl \
  --benchmark_eval_source data/benchmark.jsonl \
  --output_dir runs/chatlm-mini-sft \
  --model_family causal \
  --generation_eval_top_k 5
```

Quick wrapper for `charent/ChatLM-mini-Chinese`:

```bash
python scripts/run_chatlm_quick.py \
  --train_source data/sft.jsonl \
  --benchmark_source data/benchmark.jsonl \
  --output_dir runs/chatlm-mini-sft \
  --mode sft \
  --top_k 5
```

For a contrastive-style dataset where the prompt lives in an `anchor` field, evaluate only the anchor side plus the same benchmark:

```bash
encdec-sft \
  --model_name_or_path charent/ChatLM-mini-Chinese \
  --train_source data/sft.jsonl \
  --anchor_eval_source data/contrastive.jsonl \
  --anchor_field anchor \
  --anchor_response_field response \
  --benchmark_eval_source data/benchmark.jsonl \
  --output_dir runs/chatlm-mini-contrastive-anchor-eval \
  --model_family causal \
  --generation_eval_top_k 5
```

Quick contrastive anchor run:

```bash
python scripts/run_chatlm_quick.py \
  --train_source data/sft.jsonl \
  --anchor_source data/contrastive.jsonl \
  --benchmark_source data/benchmark.jsonl \
  --output_dir runs/chatlm-mini-anchor \
  --mode contrastive \
  --top_k 5
```

## Exact Evaluation

```bash
encdec-eval-exact \
  --model_name_or_path runs/chatlm-mini-sft \
  --eval_source examples/eval.jsonl \
  --output_path runs/chatlm-mini-eval/predictions.jsonl \
  --model_family causal \
  --top_k 5
```

The evaluator generates one or more candidate responses per prompt and compares them to the reference response. `accuracy` is exact match for the first candidate, while `top_5_accuracy` checks whether any of the five beam candidates exactly matches. By default it strips leading and trailing whitespace only. Add `--collapse_whitespace` or `--lowercase` if you want a more forgiving exact match.

When generation eval is launched from `encdec-sft`, predictions are written under `OUTPUT_DIR/generation_eval/`, with aggregate metrics in `OUTPUT_DIR/generation_eval/metrics.json`.

## Pruning

Magnitude pruning, unstructured by absolute weight value:

```bash
encdec-prune \
  --model_name_or_path runs/chatlm-mini-sft \
  --method magnitude \
  --sparsity 0.5 \
  --output_dir runs/chatlm-mini-magnitude-50
```

Gradient pruning, using the vanilla first-order score `abs(weight * gradient)` from calibration examples:

```bash
encdec-prune \
  --model_name_or_path runs/chatlm-mini-sft \
  --method gradient \
  --sparsity 0.5 \
  --calibration_source examples/train.jsonl \
  --calibration_limit 128 \
  --output_dir runs/chatlm-mini-gradient-50
```

NVIDIA 2:4 pruning, keeping the two largest-magnitude weights in every group of four input weights:

```bash
encdec-prune \
  --model_name_or_path runs/chatlm-mini-sft \
  --method nvidia \
  --nvidia_keep_n 2 \
  --nvidia_group_m 4 \
  --output_dir runs/chatlm-mini-nvidia-2-4
```

WANDA pruning, using `abs(weight) * sqrt(input_activation_mean_square)` and pruning row-wise:

```bash
encdec-prune \
  --model_name_or_path runs/chatlm-mini-sft \
  --method wanda \
  --sparsity 0.5 \
  --calibration_source examples/train.jsonl \
  --calibration_limit 128 \
  --output_dir runs/chatlm-mini-wanda-50
```

Every pruning run saves the pruned model, tokenizer, and `pruning_report.json` in the output directory. By default, linear layers are pruned and `lm_head` is skipped. Add `--include_lm_head` if you want to prune it too.

The NVIDIA method follows the usual 2:4 constraint strictly: a linear layer is pruned only when its input dimension is divisible by the group size. Non-divisible layers are skipped instead of leaving a partial dense remainder.

## Local Scripts

The package commands are also available as direct scripts:

```bash
python scripts/sft.py --help
python scripts/evaluate_exact.py --help
python scripts/prune.py --help
```

## Tests

```bash
python -m compileall src scripts tests
pytest
```
