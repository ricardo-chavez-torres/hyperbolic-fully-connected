# Hyperbolic Fully Connected Networks

PyTorch implementation of fully connected layers and ResNet architectures on the Lorentz model of hyperbolic space.

## Overview

This repository provides hyperbolic neural network layers operating on the Lorentz (hyperboloid) model, including:

- **Lorentz fully connected layers** with signed distance to hyperplanes
- **Lorentz batch normalization** via tangent space normalization with Frechet mean
- **Lorentz convolution** using direct concatenation of Lorentz points
- **Lorentz ResNet-18** for image classification
- **Lorentz multinomial logistic regression (MLR)** classifier
- Baseline implementations: Poincare ball model, Chen et al. (2022) linear layer, Bdeir et al. MLR

## Installation

Requires Python >= 3.12.

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

PyTorch with CUDA 12.6 is fetched from the PyTorch index automatically. The `geoopt` package is installed from source.

## Project Structure

```
layers/               Core hyperbolic layer implementations
  lorentz.py          Lorentz manifold operations (exp/log maps, parallel transport, etc.)
  LLinear.py          Lorentz fully connected layers and MLR classifier
  LConv.py            Lorentz 2D convolution
  LBatchNorm.py       Lorentz batch normalization (1D and 2D)
  LResBlock.py        Lorentz residual block
  LResNet.py          Lorentz ResNet-18
  LProjection.py      Euclidean-to-Lorentz input projection
  poincare.py         Poincare ball model layers
  chen.py             Chen et al. (2022) linear layer
  bdeir.py            Bdeir et al. MLR

cifar_exp/            CIFAR-10 classification experiment
  main.py             Training script with W&B sweep support

toy_exp/              Hyperplane convergence experiments
  toy_exp.py          Tests convergence of hyperplane learning at various distances

runtime_exp/          Runtime benchmarking
  my_runtimes.py      Benchmarks across layer implementations and dimensions
  runtime_results_mine_2026-03.csv            its results, March 2026
  runtime_results_mine_2026-07_titan_rtx.csv  re-run with ILNN added, July 2026, TITAN RTX
  benchmark_training_time.sh   CIFAR-100 training time of all models on one GPU

baselines/            Third-party baselines, with their licenses (see baselines/README.md)
  ilnn/               ILNN (Intrinsic Lorentz Neural Network) ResNet-18
  poincare_resnet/    Poincaré ResNet model code
```

## Usage

### CIFAR-10 Training

```bash
cd cifar_exp
python main.py
```

Training is configured through the `default_config` dictionary in `main.py` and integrates with [Weights & Biases](https://wandb.ai/) for experiment tracking and hyperparameter sweeps. Key configuration options:

| Parameter | Description | Default |
|---|---|---|
| `manifold` | `"lorentz"` or `"euclidean"` | `"lorentz"` |
| `fc_variant` | `"ours"` or `"theirs"` (Chen et al.) | `"ours"` |
| `normalisation_mode` | BN mode: `"normal"`, `"centering_only"`, etc. | `"centering_only"` |
| `optimizer` | `"adam"` or `"sgd"` (Riemannian) | `"sgd"` |
| `hidden_dim` | Base channel dimension | `64` |
| `num_epochs` | Training epochs | `200` |

### Using Layers Directly

```python
from layers import Lorentz, lorentz_resnet18, LorentzFullyConnectedOurs

manifold = Lorentz(k_value=1.0)

# ResNet-18 in Lorentz space
model = lorentz_resnet18(num_classes=10, base_dim=64, manifold=manifold)

# Single fully connected layer
fc = LorentzFullyConnectedOurs(
    in_features=65,   # 64 spatial + 1 time
    out_features=129,  # 128 spatial + 1 time
    manifold=manifold,
)
```

## Layer Variants

The `fc_variant` parameter selects between two fully connected layer implementations:

- **`"ours"`** (`LorentzFullyConnectedOurs`): Computes signed distance to learned hyperplanes in the Lorentz model. Supports weight normalization and an MLR classification mode.
- **`"theirs"`** (`LorentzFullyConnectedTheirs`): Based on Chen et al. (2022), applies a standard linear map followed by projection onto the manifold.

## Baselines

ILNN and the Poincaré ResNet are compared against using their original code, included in
`baselines/` with their licenses and notices. ILNN was modified only to use the same held-out
validation split as `cifar_exp/main.py` and to record training time. The upstream commits, every
change, and the commands that reproduce the reported numbers are listed in
[baselines/README.md](baselines/README.md).

## References

- Chen, W., Han, X., Lin, Y., Zhao, H., Liu, Z., Li, P., Sun, M., & Zhou, J. (2022). Fully Hyperbolic Neural Networks. *ACL*.
- Bdeir, A., Schwab, N., & Nicklas, D. (2024). Fully Hyperbolic Convolutional Neural Networks for Computer Vision. *ICLR*.

## License

MIT
