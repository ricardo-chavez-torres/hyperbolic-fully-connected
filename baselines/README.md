# Baselines

Third-party code used for the comparisons in the paper, included so that every number can be
reproduced from this repository alone. Each directory keeps its original license and notices.
The unmodified upstream files were committed first, and any later change was made in separate
commits, so `git log -- baselines/<name>` and `git diff` against the import commit show exactly
what was changed.

| Directory | Upstream | Commit | License | Modified |
|---|---|---|---|---|
| `ilnn/` | [Longchentong/ILNN](https://github.com/Longchentong/ILNN) (Intrinsic Lorentz Neural Network, ICLR 2026) | `7432ea7` | MIT (+ MIT for HyperbolicCV, Apache-2.0 for the vendored Geoopt; see `ilnn/NOTICE`) | yes, see below |
| `poincare_resnet/` | [maxvanspengler/poincare-resnet](https://github.com/maxvanspengler/poincare-resnet) (Poincaré ResNet, ICCV 2023) | `fbbbbf8` | Apache-2.0 | no |

Both run in this repository's environment (`uv sync` at the root); no separate installation
is needed.

## ILNN (`ilnn/`)

Only the vision part of the upstream repository is included: `lib/` (with its vendored Geoopt
snapshot), `share.py` and `experiments/vision/`. The genomics experiments, tools, tests and data
are not.

**Changes** (none touch the model, the optimizer or the training schedule):

1. **Validation protocol.** Upstream selects the best epoch on the test set. `train.py` now
   holds out 10% of the training set (`--val_fraction 0.1`, `--data_split_seed 42`) with the same
   index arithmetic as `cifar_exp/main.py`, so ILNN and our models train on the same 45k images
   and select their epoch on the same 5k. The reloaded best-validation checkpoint is tested once
   (`best_test_acc1` in `runtime.json`). `--val_fraction 0` restores the published protocol.
2. **Two upstream bugs that only matter once such a split exists.** `select_dataset()` returned
   `(train, test, val)` while `train.py` unpacked `(train, val, test)`. The validation subset was
   also drawn from the augmented training copy. Validation now uses the evaluation transform.
3. **Runtime record.** `<output_dir>/<run>/runtime.json` holds per-epoch training/validation
   time, validation loss and accuracy, and test accuracy, in the same schema as
   `cifar_exp/main.py --runtime_json`. The validation loss is averaged per sample instead of per
   batch.
4. **PyTorch ≥ 2.6.** `torch.load(..., weights_only=False)`, because checkpoints store the
   argparse namespace.
5. **Running from this repository.** `--data_dir` lets ILNN share `cifar_exp/data/cifar`.
   `lib/geoopt/LICENSE` adds the Apache-2.0 license that upstream omits for its Geoopt copy.

**CIFAR-10/100 as reported in the paper** (ResNet-18, 200 epochs, seeds 1-3; `--output_dir` and
`--data_dir` are resolved against `baselines/ilnn`, so pass absolute paths):

```bash
for SEED in 1 2 3; do
  for DS in CIFAR10 CIFAR100; do
    mkdir -p "$PWD/outputs/ilnn_${DS}_seed$SEED"   # train.py does not create parent directories
    uv run python baselines/ilnn/experiments/vision/train.py \
      -c baselines/ilnn/experiments/vision/config/ILNN-$DS.txt \
      --seed $SEED --data_dir "$PWD/cifar_exp/data/cifar" \
      --output_dir "$PWD/outputs/ilnn_${DS}_seed$SEED"
  done
done
```

Each run writes `runtime.json` next to its checkpoints; the reported accuracy is
`best_test_acc1`. A run takes about 13 h on a TITAN RTX. `experiments/vision/test.py
--mode visualize_embeddings` additionally needs `umap-learn`, which is not a dependency here.

`layers/ilnn.py` and `HypLinearILNN` in `hgcn/` are separate re-implementations of ILNN's
point-to-hyperplane layer for the single-layer, toy and graph experiments. The CIFAR numbers
come from the original code in this directory.

## Poincaré ResNet (`poincare_resnet/`)

Only `models/` (network, layers, manifold, optimizer set-up) and `cifar100/transforms.py` are
included, unmodified. They are used by `runtime_exp/poincare_resnet_timing.py`, which times the
upstream default ResNet-20 (`hyperbolic-8-16-32-resnet-20`) with the upstream optimizer defaults.
The driver mirrors `cifar_exp/main.py`: same train/validation split, batch size and data-loader
workers, and training time only. The upstream `train.py` is not used, because it loads data
without worker processes and counts the per-epoch test evaluation as training time.

## Training-time comparison

`runtime_exp/benchmark_training_time.sh` trains the Euclidean ResNet-18, FGG-LNN, HCNN, ILNN and
the Poincaré ResNet-20 for a few CIFAR-100 epochs each, eager and `torch.compile`d, back to back
on one GPU, and prints per-epoch times and ratios to the Euclidean baseline:

```bash
bash runtime_exp/benchmark_training_time.sh 4
```

With PyTorch 2.9, ILNN (a Triton compiler error) and the Poincaré ResNet (an Inductor assertion)
cannot be compiled; only their eager times are available.
