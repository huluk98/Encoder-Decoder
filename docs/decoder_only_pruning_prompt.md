# Decoder-Only Pruning Check Prompt

Copy and paste this prompt when you want another model or code reviewer to check that decoder-only pruning matches the working CMC/T5 scripts.

```text
I have working CMC/T5 pruning scripts for four methods:
- `magnitude (1).py`: per-layer magnitude pruning.
- `gradient (1).py`: gradient/Taylor pruning.
- `nvidia (1).py`: NVIDIA 2:4 structured pruning.
- `wanda.py`: WANDA activation-aware pruning.

Please adapt/check my decoder-only Hugging Face pruning code so it matches these same vanilla methods at 50% sparsity.

Target model type:
- Decoder-only causal LM, loaded with AutoModelForCausalLM.
- Example model: a local fine-tuned decoder-only checkpoint.
- Dataset records have {"prompt": "...", "response": "..."}.

General pruning requirements:
- Prune nn.Linear weight matrices only.
- Skip lm_head by default unless explicitly requested.
- Use per-layer pruning by default, not global pruning, because the CMC scripts prune each layer separately.
- Save the pruned model/tokenizer and write a pruning_report.json with total sparsity and per-layer sparsity.
- Use 50% sparsity for magnitude, gradient, and WANDA.
- For decoder-only calibration, concatenate prompt + response, but set labels to -100 for prompt tokens so loss/gradients are computed only on response tokens.

Magnitude pruning:
- For every Linear layer, compute score = abs(weight).
- Flatten scores per layer.
- Keep top 50% highest scores.
- Zero the bottom 50%.
- This should match the CMC magnitude script behavior.

Gradient/Taylor pruning:
- Run calibration batches through the decoder-only LM using response-only labels.
- Backpropagate loss.
- For every Linear layer, accumulate score += abs(weight * gradient).
- After calibration, per layer keep top 50% scores and zero the bottom 50%.
- Reset gradients between batches.
- This should match the CMC gradient/Taylor script, adapted for causal LM.

NVIDIA 2:4 pruning:
- For every Linear layer with input dimension divisible by 4, reshape weight as [out_features, num_groups, 4].
- In each group of 4, zero the 2 smallest absolute values and keep the 2 largest.
- This creates exactly 50% sparsity in eligible layers.
- If input dimension is not divisible by 4, skip that layer entirely.
- Do not prune a partial remainder group.
- This should match the CMC NVIDIA script.

WANDA pruning:
- Run calibration prompts through the decoder-only LM and collect input activations to each Linear layer.
- For each Linear layer, compute activation norm per input dimension.
- Compute score = abs(weight) * activation_norm.
- Prune per row: for each output row, keep the top 50% input columns by score and zero the bottom 50%.
- This should match official WANDA behavior and the CMC WANDA script.

Please verify:
- Magnitude reaches about 50% sparsity per pruned Linear layer.
- Gradient reaches about 50% sparsity per pruned Linear layer.
- WANDA reaches about 50% sparsity per pruned Linear layer row-wise.
- NVIDIA reaches 2:4 sparsity only for eligible layers and skips non-divisible layers.
- The code works for decoder-only models, not just T5/encoder-decoder models.
```

## Method Summary

- Magnitude removes the smallest weights by absolute value.
- Gradient/Taylor removes weights with the smallest `abs(weight * gradient)`, meaning weights estimated to matter least to the calibration loss.
- NVIDIA 2:4 keeps two weights and zeros two weights in every group of four input weights.
- WANDA removes weights using both weight size and input activation strength: `abs(weight) * activation_norm`, pruning row-wise.
