# Reference CMC Pruning Scripts

These files are the original working CMC/T5 pruning scripts provided for comparison:

- `magnitude_cmc.py`: per-layer magnitude pruning at 50% sparsity.
- `gradient_cmc.py`: Taylor/gradient pruning with `abs(weight * gradient)`.
- `nvidia_cmc.py`: strict NVIDIA 2:4 pruning.
- `wanda_cmc.py`: WANDA pruning with row-wise 50% sparsity.

The maintained, configurable implementations used by the training suite live in
`src/encoder_decoder/pruning.py` and are launched by `scripts/run_pruning_eval.sh`.
