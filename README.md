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

If you already have a Python environment and only need the packages:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

Verify the environment before launching training:

```bash
python -c "import torch, transformers, datasets, accelerate; print('env ok')"
```

See [docs/decoder_only_pruning_prompt.md](docs/decoder_only_pruning_prompt.md) for a copy-paste prompt that describes the expected decoder-only 50% pruning behavior against the CMC scripts.

## Copy-Paste Quickstart

Install and enter the environment:

```bash
git clone https://github.com/huluk98/Encoder-Decoder.git
cd Encoder-Decoder
conda env create -f environment.yml
conda activate encoder-decoder-prune
pip install -e ".[dev]"
python -c "import torch, transformers, datasets, accelerate; print('env ok')"
```

Run a tiny smoke test with the included examples:

```bash
python scripts/run_chatlm_quick.py \
  --train_source examples/train.jsonl \
  --benchmark_source examples/eval.jsonl \
  --output_dir runs/smoke-chatlm \
  --mode sft \
  --max_steps 1 \
  --generation_eval_limit 1 \
  --top_k 5
```

Run regular SFT on your SFT dataset and benchmark:

```bash
python scripts/run_chatlm_quick.py \
  --train_source data/sft.jsonl \
  --benchmark_source data/benchmark.jsonl \
  --output_dir runs/chatlm-mini-sft \
  --mode sft \
  --model_family seq2seq \
  --num_train_epochs 3 \
  --learning_rate 5e-5 \
  --top_k 5
```

Run 8-GPU regular SFT with top-1 and top-5 exact eval:

1. Edit the path block at the top of [scripts/train_sft_8gpu.py](scripts/train_sft_8gpu.py):

```python
MODEL_NAME_OR_PATH = "charent/ChatLM-mini-Chinese"
TRAIN_SOURCE = "data/sft.jsonl"
TRAIN_EVAL_SOURCE = TRAIN_SOURCE
BENCHMARK_SOURCE = "data/benchmark.jsonl"
OUTPUT_DIR = "runs/chatlm-mini-8gpu-sft"
NUM_TRAIN_EPOCHS = 3.0
PRECISION = "bf16"  # use "bf16", "fp16", or "fp32"
```

2. Launch SFT training and evaluation:

```bash
python scripts/train_sft_8gpu.py
```

3. Read the top-1/top-5 output:

```bash
cat runs/chatlm-mini-8gpu-sft/generation_eval/top1_top5_metrics.json
```

For a quick command preview without launching:

```bash
python scripts/train_sft_8gpu.py --dry_run
```

You can also override the paths without editing the file:

```bash
python scripts/train_sft_8gpu.py \
  --model_path charent/ChatLM-mini-Chinese \
  --train_source /path/to/sft.jsonl \
  --train_eval_source /path/to/sft.jsonl \
  --benchmark_source /path/to/benchmark.jsonl \
  --output_dir runs/chatlm-mini-8gpu-sft \
  --epochs 3 \
  --precision bf16
```

Use `--precision fp16` instead if your GPUs/checkpoint run better in float16.

Run the same one-command training/eval flow on physical GPUs `4,5,6,7`:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n DPO \
  python scripts/train_sft_8gpu.py \
  --nproc_per_node 4 \
  --model_path charent/ChatLM-mini-Chinese \
  --train_source data/sft.jsonl \
  --train_eval_source data/sft.jsonl \
  --benchmark_source data/benchmark.jsonl \
  --output_dir runs/chatlm-mini-4gpu-sft \
  --epochs 3 \
  --precision bf16
```

For contrastive SFT plus eval on physical GPUs `4,5,6,7`:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n DPO \
  python scripts/train_contrastive_8gpu.py \
  --nproc_per_node 4 \
  --model_path charent/ChatLM-mini-Chinese \
  --contrastive_train_source "/Users/luke/Documents/SCENIC agent/data/SCENIC_full_anchor_positive_negative.json" \
  --sft_train_eval_source data/sft.jsonl \
  --benchmark_source data/benchmark.jsonl \
  --output_dir runs/chatlm-mini-4gpu-contrastive \
  --negative_field invalid_negative \
  --epochs 3 \
  --precision bf16
```

When `CUDA_VISIBLE_DEVICES=4,5,6,7` is set, those physical GPUs are exposed to the process as local CUDA devices `0,1,2,3`. Use `--nproc_per_node 4`, not `8`, when exposing four GPUs.

You can also run the same 4-GPU commands from a shell launcher instead of typing the full CLI:

```bash
./scripts/run_4gpu_train_eval.sh sft
./scripts/run_4gpu_train_eval.sh contrastive
./scripts/run_4gpu_train_eval.sh both
```

Preview the generated commands without launching training:

```bash
DRY_RUN=1 ./scripts/run_4gpu_train_eval.sh both
```

The launcher uses `CUDA_VISIBLE_DEVICES=4,5,6,7`, `NPROC_PER_NODE=4`, and the `DPO` conda environment by default. Change file paths and JSON paths in the `EDIT THIS BLOCK` section at the top of [scripts/run_4gpu_train_eval.sh](scripts/run_4gpu_train_eval.sh):

```bash
DEFAULT_TRAIN_SOURCE="data/sft.jsonl"
DEFAULT_TRAIN_EVAL_SOURCE="${DEFAULT_TRAIN_SOURCE}"
DEFAULT_BENCHMARK_SOURCE="data/benchmark.jsonl"
DEFAULT_CONTRASTIVE_TRAIN_SOURCE="/Users/luke/Documents/SCENIC agent/data/SCENIC_full_anchor_positive_negative.json"
DEFAULT_SFT_TRAIN_EVAL_SOURCE="data/sft.jsonl"
DEFAULT_CONTRASTIVE_BENCHMARK_SOURCE="${DEFAULT_BENCHMARK_SOURCE}"
```

You can still override paths or training settings without editing the file by using environment variables:

```bash
TRAIN_SOURCE=data/my_sft.jsonl \
BENCHMARK_SOURCE=data/my_benchmark.jsonl \
EPOCHS=1 \
./scripts/run_4gpu_train_eval.sh sft
```

If you see `EADDRINUSE` or `address already in use` for port `29500`, another distributed job is already using torchrun's default port. The launcher defaults to `MASTER_PORT=29573`; choose another free port if needed:

```bash
MASTER_PORT=29601 ./scripts/run_4gpu_train_eval.sh sft
MASTER_PORT=29602 ./scripts/run_4gpu_train_eval.sh contrastive
```

If the job is running but you do not see it by name, remember that torchrun workers often appear as `python` or `sft.py`, not as the launcher script. Check the process, port, and GPU usage with:

```bash
./scripts/check_training_process.sh
MASTER_PORT=29601 ./scripts/check_training_process.sh
```

Run 8-GPU contrastive SFT with top-1 and top-5 exact eval:

1. Edit the path block at the top of [scripts/train_contrastive_8gpu.py](scripts/train_contrastive_8gpu.py):

```python
MODEL_NAME_OR_PATH = "charent/ChatLM-mini-Chinese"
CONTRASTIVE_TRAIN_SOURCE = "/Users/luke/Documents/SCENIC agent/data/SCENIC_full_anchor_positive_negative.json"
SFT_TRAIN_EVAL_SOURCE = "data/sft.jsonl"
BENCHMARK_SOURCE = "data/benchmark.jsonl"
OUTPUT_DIR = "runs/chatlm-mini-8gpu-contrastive"
NEGATIVE_FIELD = "invalid_negative"
```

2. Launch contrastive training and evaluation:

```bash
python scripts/train_contrastive_8gpu.py
```

3. Read the top-1/top-5 output:

```bash
cat runs/chatlm-mini-8gpu-contrastive/generation_eval/top1_top5_metrics.json
```

For a quick command preview:

```bash
python scripts/train_contrastive_8gpu.py --dry_run
```

The contrastive script trains on `anchor`, `positive`, and `negative`/`invalid_negative`, then evaluates the trained model on the regular SFT training data and the benchmark. To use valid hard negatives instead of invalid compatibility negatives, set:

```python
NEGATIVE_FIELD = "negative"
```

Contrastive mode follows compatibility-aware triplet SFT: it trains generation loss on both `anchor -> response` and `positive -> response`, then adds a triplet alignment loss that pulls the anchor representation toward `positive` and pushes it away from `negative`. The direct script uses `NEGATIVE_FIELD = "invalid_negative"` by default; pass `--negative_field negative` if you want valid hard negatives instead.

After each run, check:

```bash
cat runs/chatlm-mini-8gpu-sft/generation_eval/top1_top5_metrics.json
cat runs/chatlm-mini-8gpu-contrastive/generation_eval/top1_top5_metrics.json
```

## Data Format

Local `.jsonl`, `.json`, `.csv`, `.tsv`, Hugging Face datasets, and `datasets.save_to_disk(...)` directories are supported. The default columns are `prompt` and `response`.

```json
{"prompt":"Summarize: Transformers are sequence models.","response":"Transformers model sequences with attention."}
```

Use `--prompt_field` and `--response_field` if your columns differ.

Contrastive triplet SFT expects anchor-positive-negative records. The SCENIC file already follows this shape:

```json
{
  "anchor": "10点关空调。",
  "positive": "请在10点关闭空调。",
  "negative": "空调1点半关。",
  "invalid_negative": "把空调亮度调高。",
  "response": "好的，已为空调设置10点关闭。"
}
```

Use `--negative_field negative` in `scripts/train_contrastive_8gpu.py` for valid hard negatives with different responses, or `--negative_field invalid_negative` for the invalid compatibility negative shown in the triplet algorithm. The lower-level `encdec-sft` command exposes the same choice as `--contrastive_negative_field`.

## SFT

ChatLM-mini example:

```bash
encdec-sft \
  --model_name_or_path charent/ChatLM-mini-Chinese \
  --train_source examples/train.jsonl \
  --eval_source examples/eval.jsonl \
  --output_dir runs/chatlm-mini-sft \
  --model_family seq2seq \
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
  --model_family seq2seq \
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

For a contrastive-style dataset where the prompt lives in an `anchor` field, train with triplet SFT and evaluate only the anchor side plus the same benchmark:

```bash
encdec-sft \
  --model_name_or_path charent/ChatLM-mini-Chinese \
  --train_source data/contrastive.jsonl \
  --training_mode contrastive \
  --contrastive_anchor_field anchor \
  --contrastive_positive_field positive \
  --contrastive_negative_field invalid_negative \
  --contrastive_response_field response \
  --contrastive_loss_weight 0.2 \
  --contrastive_margin 0.2 \
  --anchor_eval_source data/contrastive.jsonl \
  --anchor_field anchor \
  --anchor_response_field response \
  --benchmark_eval_source data/benchmark.jsonl \
  --output_dir runs/chatlm-mini-contrastive-anchor-eval \
  --model_family seq2seq \
  --generation_eval_top_k 5
```

Quick contrastive triplet run:

```bash
python scripts/run_chatlm_quick.py \
  --train_source data/contrastive.jsonl \
  --sft_eval_source data/sft.jsonl \
  --anchor_source data/contrastive.jsonl \
  --benchmark_source data/benchmark.jsonl \
  --output_dir runs/chatlm-mini-contrastive \
  --mode contrastive \
  --contrastive_negative_field invalid_negative \
  --contrastive_response_field response \
  --top_k 5
```

## Exact Evaluation

Evaluate one saved model on a JSON/JSONL eval file and a benchmark file:

```bash
python scripts/eval_model_and_benchmark.py \
  --model_path runs/chatlm-mini-8gpu-sft \
  --eval_source data/sft.jsonl \
  --benchmark_source data/benchmark.jsonl \
  --output_dir runs/eval/chatlm-mini-8gpu-sft \
  --model_family seq2seq \
  --precision bf16 \
  --top_k 5
```

For a local checkpoint, pass the saved model directory, not a single weight file. Absolute paths are safest, and paths with spaces must be quoted:

```bash
python scripts/eval_model_and_benchmark.py \
  --model_path "/absolute/path/to/runs/chatlm-mini-8gpu-sft" \
  --eval_source "/absolute/path/to/eval.json" \
  --benchmark_source "/absolute/path/to/benchmark.json" \
  --output_dir runs/eval/local-model \
  --model_family seq2seq \
  --precision bf16 \
  --top_k 5
```

If you pass a training output directory that has no final `config.json` but does contain `checkpoint-*` folders, the loader uses the latest checkpoint automatically.

If your eval file uses SCENIC anchor fields while the benchmark uses prompt fields:

```bash
python scripts/eval_model_and_benchmark.py \
  --model_path runs/chatlm-mini-8gpu-contrastive \
  --eval_source "/Users/luke/Documents/SCENIC agent/data/SCENIC_full_anchor_positive_negative.json" \
  --eval_prompt_field anchor \
  --eval_response_field response \
  --benchmark_source data/benchmark.jsonl \
  --benchmark_prompt_field prompt \
  --benchmark_response_field response \
  --output_dir runs/eval/chatlm-mini-8gpu-contrastive \
  --model_family seq2seq \
  --precision bf16 \
  --top_k 5
```

The wrapper writes `eval_metrics.json`, `benchmark_metrics.json`, prediction JSONL files, and `top1_top{K}_metrics.json` under `--output_dir` (`top1_top5_metrics.json` with the default `--top_k 5`).

Alternative ChatLM/T5-style evaluator with textual `[EOS]` prompt handling:

Edit the path block at the top of [scripts/eval_chatlm_eos.py](scripts/eval_chatlm_eos.py), then run it directly:

```bash
python scripts/eval_chatlm_eos.py
```

The defaults in that file are:

```python
MODEL_PATH = "/Users/luke/Documents/Encoder-Decoder/runs/chatlm-mini-8gpu-sft"
BASE_MODEL_PATH = "charent/ChatLM-mini-Chinese"
EVAL_FILE = "/Users/luke/Documents/SCENIC agent/data/SCENIC_full_training_dataset.json"
BENCHMARK_FILE = "/Users/luke/Documents/SCENIC agent/generated/iot_instruction_benchmark_200.json"
OUTPUT_DIR = "/Users/luke/Documents/Encoder-Decoder/runs/eval/chatlm-eos"
USE_8_GPUS = True
CUDA_VISIBLE_DEVICES = "4,5,6,7"
NPROC_PER_NODE = 4
BATCH_SIZE = 256  # per GPU on H20; effective 1024 prompts across GPUs 4-7
MIN_BATCH_SIZE = 1  # fallback if a long batch causes CUDA OOM
```

If your local checkpoint says it cannot find `modeling_chat_model.py`, keep `BASE_MODEL_PATH = "charent/ChatLM-mini-Chinese"`. The script will load the custom ChatLM architecture from that base model and then apply the weights from `MODEL_PATH`, so the checkpoint folder itself does not need to contain `modeling_chat_model.py`. If the machine has no internet, set `BASE_MODEL_PATH` to a local copy of `charent/ChatLM-mini-Chinese` that contains the custom code files.

For regular `encdec-*` commands, newly saved SFT/pruned checkpoints copy the required ChatLM custom-code files automatically. To repair an older checkpoint that is already missing `modeling_chat_model.py`, run:

```bash
python scripts/repair_custom_code.py \
  --checkpoint_dir runs/chatlm-mini-4gpu-sft \
  --base_model_name_or_path charent/ChatLM-mini-Chinese
```

With `USE_8_GPUS = True`, a normal `python scripts/eval_chatlm_eos.py` relaunches itself with `CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4`, splits eval rows across those GPUs, and merges predictions/metrics into `OUTPUT_DIR/summary.json`. Use `--no_8_gpus` to force single-GPU evaluation.

You can still override any path from the command line:

```bash
python scripts/eval_chatlm_eos.py \
  --model_path "/absolute/path/to/ChatLM-mini-Chinese-or-checkpoint" \
  --eval_file "/absolute/path/to/eval.json" \
  --benchmark_file "/absolute/path/to/benchmark.json" \
  --output_dir runs/eval/chatlm-eos \
  --top_k 5 \
  --batch_size 256
```

This separate script accepts JSON, JSONL, and wrappers such as `{"records": [...]}`. It auto-detects prompt keys like `prompt`, `input`, `question`, or `instruction`, and response keys like `response`, `answer`, `output`, or `target`. Use `--prompt_key anchor --response_key response` for SCENIC anchor files.

For SCENIC anchor eval plus a normal prompt/response benchmark:

```bash
python scripts/eval_chatlm_eos.py \
  --model_path "/absolute/path/to/model_or_checkpoint" \
  --eval_file "/Users/luke/Documents/SCENIC agent/data/SCENIC_full_anchor_positive_negative.json" \
  --prompt_key anchor \
  --response_key response \
  --benchmark_file "/absolute/path/to/benchmark.json" \
  --benchmark_prompt_key prompt \
  --benchmark_response_key response \
  --output_dir runs/eval/chatlm-eos-anchor \
  --top_k 5
```

```bash
encdec-eval-exact \
  --model_name_or_path runs/chatlm-mini-sft \
  --eval_source examples/eval.jsonl \
  --output_path runs/chatlm-mini-eval/predictions.jsonl \
  --model_family seq2seq \
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

Run all four pruning methods after regular SFT and report exact match plus exact@5 on the SFT training data and benchmark:

```bash
MODEL_PATH=runs/chatlm-mini-8gpu-sft \
CALIBRATION_SOURCE=data/sft.jsonl \
EVAL_SOURCE=data/sft.jsonl \
BENCHMARK_SOURCE=data/benchmark.jsonl \
OUTPUT_ROOT=runs/pruning_eval/sft \
MODEL_FAMILY=seq2seq \
PRECISION=bf16 \
SPARSITY=0.5 \
TOP_K=5 \
METHODS="magnitude gradient nvidia wanda" \
bash scripts/run_pruning_eval.sh
```

Run the same four methods after contrastive SFT, calibrating WANDA/gradient on the contrastive anchors and evaluating on the regular SFT training data plus the benchmark:

```bash
MODEL_PATH=runs/chatlm-mini-8gpu-contrastive \
CALIBRATION_SOURCE="/Users/luke/Documents/SCENIC agent/data/SCENIC_full_anchor_positive_negative.json" \
CALIBRATION_PROMPT_FIELD=anchor \
CALIBRATION_RESPONSE_FIELD=response \
EVAL_SOURCE=data/sft.jsonl \
EVAL_PROMPT_FIELD=prompt \
EVAL_RESPONSE_FIELD=response \
BENCHMARK_SOURCE=data/benchmark.jsonl \
BENCHMARK_PROMPT_FIELD=prompt \
BENCHMARK_RESPONSE_FIELD=response \
OUTPUT_ROOT=runs/pruning_eval/contrastive \
MODEL_FAMILY=seq2seq \
PRECISION=bf16 \
SPARSITY=0.5 \
TOP_K=5 \
METHODS="magnitude gradient nvidia wanda" \
bash scripts/run_pruning_eval.sh
```

The script runs the four CMC-style methods you provided:

- `magnitude`: matches `magnitude (1).py`, using per-layer `abs(weight)` scores.
- `gradient`: matches `gradient (1).py`, using Taylor scores `abs(weight * gradient)` on calibration examples.
- `nvidia`: matches `nvidia (1).py`, using strict NVIDIA 2:4 pruning.
- `wanda`: matches `wanda.py`, using `abs(weight) * activation_norm` and row-wise pruning.

Set `PRECISION=fp16` if the checkpoint should load in float16 instead of bf16. If your JSON fields are not named `prompt` and `response`, add `PROMPT_FIELD=...` and `RESPONSE_FIELD=...`. For example:

```bash
MODEL_PATH=/path/to/t5-or-chatlm-sft \
CALIBRATION_SOURCE=/path/to/sft.jsonl \
EVAL_SOURCE=/path/to/eval.jsonl \
BENCHMARK_SOURCE=/path/to/benchmark.jsonl \
PROMPT_FIELD=instruction \
RESPONSE_FIELD=output \
PRECISION=fp16 \
bash scripts/run_pruning_eval.sh
```

It prints a summary table:

```text
method    split       top1_exact    exact@5    total
```

Full outputs are saved under `runs/pruning_eval/sft/METHOD/` and `runs/pruning_eval/contrastive/METHOD/`, including pruned models, prediction JSONL files, `eval_metrics.json`, and `benchmark_metrics.json`.

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
