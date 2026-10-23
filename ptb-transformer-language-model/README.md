# PTB Transformer Language Model

A causal Transformer language model trained on the Penn Treebank (PTB) corpus, with a hyperparameter sweep over the number of attention heads and encoder layers, evaluated by perplexity (PPL).

## Model

Embedding + sinusoidal positional encoding + `TransformerEncoder` (causal mask) + tied input/output softmax. Trained with AdamW, a linear warmup followed by cosine LR decay, label smoothing, and gradient clipping. Early stopping is based on validation perplexity.

| Hyperparameter | Default |
|---|---|
| Embedding dim | 256 |
| FFN dim | 512 |
| Dropout | 0.3 |
| Context length | 256 |
| Stride | 16 |
| Batch size | 64 |
| Learning rate | 1e-3 |
| Epochs | 50 (with early stopping, patience 10) |

`EMBED_DIM` must be divisible by `n_heads` (default 256 → heads ∈ {1, 2, 4, 8, 16, 32}).

## Dataset

Word-level [Penn Treebank](https://catalog.ldc.upenn.edu/LDC99T42) (PTB), the standard small-scale language modeling benchmark (~10K vocabulary). Included under [`data/`](data/):

```
data/ptb.train.txt
data/ptb.valid.txt
data/ptb.test.txt
```

## Running

```bash
pip install torch torchtext matplotlib tqdm
```

```bash
# Full 3x3 grid: heads in {2,4,8}, layers in {2,4,8}
python train.py --mode all

# Single run
python train.py --mode single --heads 4 --layers 8

# Batch of specific configs
python train.py --mode batch --heads 2 4 8 --layers 2 4 8

# Custom hyperparameters
python train.py --mode single --heads 8 --layers 4 \
  --epochs 50 --batch-size 64 --lr 1e-3 \
  --context-length 256 --stride 16 --dropout 0.3
```

### Key arguments

| Flag | Default | Description |
|---|---|---|
| `--mode` | `single` | `single`, `batch`, or `all` |
| `--heads` / `--layers` | — | one or more values; required for `single`/`batch` |
| `--epochs` | 50 | training epochs |
| `--batch-size` | 64 | |
| `--lr` | 1e-3 | |
| `--embed-dim` | 256 | |
| `--ffn-dim` | 512 | |
| `--dropout` | 0.3 | |
| `--context-length` | 256 | sliding window size |
| `--stride` | 16 | sliding window stride |

### Outputs

- `checkpoints/model_h{H}_l{L}.pt` — best checkpoint by validation PPL
- `results/result_h{H}_l{L}.json` — per-run results; `results/all_results.json` for batch/all runs
- `plots/training_h{H}_l{L}.png` — train/valid/test PPL vs. epoch

`results/plots/` in this repo holds the training-curve plots captured from the original run (one per heads/layers configuration).

## Results

Full 3×3 grid, sorted by test perplexity (lower is better):

| Heads | Layers | Train PPL | Valid PPL | Test PPL | Time (min) |
|---|---|---|---|---|---|
| 4 | 8 | 28.32 | 109.25 | **103.80** | 44.7 |
| 8 | 8 | 32.00 | 108.91 | 104.00 | 41.4 |
| 2 | 8 | 36.52 | 112.37 | 106.55 | 39.0 |
| 8 | 4 | 39.31 | 123.06 | 115.13 | 37.7 |
| 4 | 4 | 39.55 | 124.95 | 117.12 | 36.0 |
| 2 | 4 | 42.54 | 133.38 | 123.26 | 35.9 |
| 8 | 2 | 63.55 | 160.37 | 147.82 | 27.0 |
| 4 | 2 | 68.15 | 167.15 | 153.16 | 23.6 |
| 2 | 2 | 73.78 | 173.03 | 159.97 | 20.9 |

**Best configuration:** 4 heads, 8 layers — test PPL 103.80. Depth (number of layers) matters more than width (number of heads): every 8-layer config beats every 4-layer config, which beats every 2-layer config, regardless of head count. Returns diminish past 4 heads at a given depth.

## Files

- `ptb_transformer_language_model.ipynb` — experiment driver, result aggregation, and plots
- `train.py` — standalone training script (dataset, model, training loop, CLI)
- `data/` — PTB word-level corpus (train/valid/test)
- `results/plots/` — training-curve plots per configuration
