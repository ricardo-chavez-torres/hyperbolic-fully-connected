"""Times training epochs of the Poincaré ResNet baseline on CIFAR-100.

The model, optimizer and augmentation code are imported unchanged from the official
repository (van Spengler et al., ICCV 2023; https://github.com/maxvanspengler/poincare-resnet),
of which baselines/poincare_resnet holds an unmodified copy (see baselines/README.md).
Its own train.py is not used for timing because it loads data in the
main process (num_workers=0) and counts the per-epoch test evaluation as epoch time. This
driver instead mirrors cifar_exp/main.py: the same 45k/5k train/validation split
(numpy RandomState(42)), batch size, DataLoader workers and pinned memory, CUDA-synchronised
training-only epoch times, and the same runtime.json schema, so the numbers can be put next
to the other models' records.

--loader_only skips the model and times just the data pipeline (same dataset, transforms and
DataLoader settings, batches copied to the GPU), i.e. the fastest an epoch could possibly be.
"""
import argparse
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--repo", default=str(REPO_ROOT / "baselines" / "poincare_resnet"),
                    help="Copy of maxvanspengler/poincare-resnet (models/ and cifar100/transforms.py)")
parser.add_argument("--model", default="hyperbolic-8-16-32-resnet-20",
                    help="Model name in the repo's convention (its default ResNet-20)")
parser.add_argument("--data_dir", default=str(REPO_ROOT / "cifar_exp" / "data" / "cifar"))
parser.add_argument("--num_epochs", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--val_fraction", type=float, default=0.1)
parser.add_argument("--data_split_seed", type=int, default=42)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--compile", action="store_true", help="torch.compile the model")
parser.add_argument("--loader_only", action="store_true", help="Time the data pipeline without a model")
# Optimizer settings: the repo's train.py defaults (RiemannianSGD, lr 1e-3, momentum 0.9, wd 1e-4).
parser.add_argument("--opt", default="sgd", choices=["sgd", "adam"])
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--exp_name", default=None)
parser.add_argument("--runtime_json", default=None)


def load_module(path, name):
    # Loaded by file path: importing the repo's cifar100 package would read its config.ini.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda"
    repo = Path(args.repo)
    sys.path.insert(0, str(repo))

    transforms = load_module(repo / "cifar100" / "transforms.py", "poincare_cifar100_transforms")
    train_set = torchvision.datasets.CIFAR100(
        args.data_dir, train=True, download=True,
        transform=transforms.get_standard_transform(train=True))

    # Same split arithmetic as cifar_exp/main.py:get_dataloaders().
    indices = list(range(len(train_set)))
    np.random.RandomState(args.data_split_seed).shuffle(indices)
    train_indices = indices[int(len(train_set) * args.val_fraction):]
    train_loader = DataLoader(Subset(train_set, train_indices), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)

    runtime = {
        "exp_name": args.exp_name or ("loader-only" if args.loader_only else "Poincare-" + args.model),
        "dataset": "cifar100",
        "seed": args.seed,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "encoder_manifold": None if args.loader_only else "poincare",
        "model_name": None if args.loader_only else args.model,
        "compile": args.compile,
        "num_workers": args.num_workers,
        "gpu_name": torch.cuda.get_device_name(device),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "train_set_size": len(train_loader.dataset),
        "iterations_per_epoch": len(train_loader),
        "epochs": [],
    }

    if not args.loader_only:
        from models.optimizers import initialize_optimizer
        from models.resnets import parse_model_from_name

        model = parse_model_from_name(args.model, 100).to(device)
        runtime["num_params"] = sum(p.numel() for p in model.parameters())
        optimizer = initialize_optimizer(model=model, args=args)
        if args.compile:
            model = torch.compile(model)
        loss_fn = torch.nn.CrossEntropyLoss()
        model.train()

    for epoch in range(args.num_epochs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if args.loader_only:
                continue
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - start
        runtime["epochs"].append({"epoch": epoch + 1, "train_time_s": epoch_time})
        print("[timing] epoch {}/{}: train={:.1f}s".format(epoch + 1, args.num_epochs, epoch_time),
              flush=True)

    times = [e["train_time_s"] for e in runtime["epochs"]]
    steady = times[1:] if len(times) > 1 else times
    runtime.update({
        "epochs_completed": len(times),
        "train_time_s": sum(times),
        "mean_epoch_train_time_s": sum(times) / len(times),
        "median_epoch_train_time_s": statistics.median(steady),
        "first_epoch_train_time_s": times[0],
        "train_time_per_1k_iters_s": 1000.0 * statistics.median(steady) / len(train_loader),
    })
    if args.runtime_json:
        Path(args.runtime_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.runtime_json).write_text(json.dumps(runtime, indent=2))
        print("[timing] runtime record written to " + args.runtime_json)


if __name__ == "__main__":
    main()
