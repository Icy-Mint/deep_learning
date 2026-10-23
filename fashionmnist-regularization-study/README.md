# FashionMNIST Regularization Study

A comparison of four regularization strategies on a LeNet-5 convolutional network trained to classify [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) images: a plain baseline, dropout, weight decay (L2), and batch normalization.

## Model

LeNet-5: two conv+pool blocks (1→6→16 channels, 5×5 kernels) followed by three fully connected layers (256→120→84→10). Each variant modifies this base architecture or its optimizer:

| Variant | Change |
|---|---|
| Baseline | Plain LeNet-5, Adam optimizer |
| Dropout | `p=0.5` dropout after `fc1` and `fc2` |
| Weight Decay | Adam with `weight_decay=1e-4` |
| Batch Normalization | `BatchNorm2d`/`BatchNorm1d` after each conv and FC layer (pre-activation) |

All variants train for 30 epochs with Adam (`lr=1e-3`), batch size 64, cross-entropy loss.

## Dataset

[FashionMNIST](https://github.com/zalandoresearch/fashion-mnist): 60,000 training / 10,000 test grayscale 28×28 images across 10 clothing categories. Downloaded automatically by `torchvision.datasets.FashionMNIST` on first run (not stored in this repo).

## Running

```bash
pip install torch torchvision matplotlib
jupyter notebook cnn_regularization_comparison.ipynb
```

Each model's cell is self-contained: it builds the dataset/loaders, trains for 30 epochs while tracking train/test accuracy, saves the trained weights to a `.pth` file, and saves a convergence plot (`*_convergence_graph.png`).

## Results

| Model | Final Train Accuracy | Final Test Accuracy |
|---|---|---|
| Baseline | 96.81% | 90.04% |
| Dropout | 91.75% | 89.01% |
| Weight Decay | 95.98% | 89.99% |
| Batch Normalization | 97.85% | 89.59% |

The baseline model already generalizes reasonably well on FashionMNIST; dropout reduces training accuracy the most (strongest regularization effect) but doesn't improve test accuracy here, while weight decay and batch norm have a milder effect on the train/test gap. Batch normalization reaches the highest training accuracy and also speeds up convergence (see the convergence graphs).

## Files

- `cnn_regularization_comparison.ipynb` — training/evaluation code for all four variants
- `*_model.pth` — saved weights for each trained model
- `*_convergence_graph.png` — train/test accuracy vs. epoch for each variant
